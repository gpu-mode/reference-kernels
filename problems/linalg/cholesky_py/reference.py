import torch

from task import input_t, output_t


_RECON_RTOL_FACTOR = 20.0
_TRIANGULAR_RTOL_FACTOR = 8.0


def _symmetrize(a: torch.Tensor) -> torch.Tensor:
    return 0.5 * (a + a.transpose(-1, -2))


def _random_orthogonal(
    batch: int,
    n: int,
    gen: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    x = torch.randn((batch, n, n), device=device, dtype=torch.float32, generator=gen)
    q, r = torch.linalg.qr(x)
    signs = torch.where(
        torch.diagonal(r, dim1=-2, dim2=-1) >= 0,
        torch.ones((), device=device),
        -torch.ones((), device=device),
    )
    return q * signs.unsqueeze(-2)


def _dense_spd(a: torch.Tensor, damping: float) -> torch.Tensor:
    n = a.shape[-1]
    out = (a @ a.transpose(-1, -2)) / float(max(n, 1))
    out.diagonal(dim1=-2, dim2=-1).add_(damping)
    return _symmetrize(out).contiguous()


def generate_input(
    batch: int,
    n: int,
    cond: int,
    seed: int,
    case: str = "dense",
) -> input_t:
    assert batch > 0, "batch must be positive"
    assert n > 0, "n must be positive"
    assert cond >= 0, "cond must be non-negative"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    case = case.lower()

    if case == "dense":
        a = torch.randn(
            (batch, n, n), device=device, dtype=torch.float32, generator=gen
        )
        return _dense_spd(a, 10.0 ** -max(cond, 2))

    if case == "diagonal":
        diagonal = torch.logspace(
            0.0,
            -float(max(cond, 1)),
            n,
            device=device,
            dtype=torch.float32,
        )
        return torch.diag_embed(diagonal.expand(batch, n)).contiguous()

    if case == "spectrum":
        values = torch.logspace(
            0.0,
            -float(max(cond, 1)),
            n,
            device=device,
            dtype=torch.float32,
        ).expand(batch, n)
        q = _random_orthogonal(batch, n, gen, device)
        return _symmetrize(
            (q * values.unsqueeze(-2)) @ q.transpose(-1, -2)
        ).contiguous()

    if case == "lowrank":
        rank = max(1, min(n // 8, 64))
        factors = torch.randn(
            (batch, n, rank),
            device=device,
            dtype=torch.float32,
            generator=gen,
        )
        out = (factors @ factors.transpose(-1, -2)) / float(rank)
        out.diagonal(dim1=-2, dim2=-1).add_(10.0 ** -max(cond, 3))
        return _symmetrize(out).contiguous()

    if case == "rowscale":
        a = torch.randn(
            (batch, n, n), device=device, dtype=torch.float32, generator=gen
        )
        base = _dense_spd(a, 1.0e-3)
        scales = torch.logspace(
            0.0,
            -0.5 * float(max(cond, 2)),
            n,
            device=device,
            dtype=torch.float32,
        )
        out = scales.reshape(1, n, 1) * base * scales.reshape(1, 1, n)
        out.diagonal(dim1=-2, dim2=-1).add_(1.0e-6)
        return _symmetrize(out).contiguous()

    if case == "tridiagonal":
        diagonal = torch.empty((batch, n), device=device, dtype=torch.float32).uniform_(
            1.5, 2.5, generator=gen
        )
        off_diagonal = torch.empty(
            (batch, max(n - 1, 0)), device=device, dtype=torch.float32
        ).uniform_(-0.25, 0.25, generator=gen)
        out = torch.diag_embed(diagonal)
        if n > 1:
            out = (
                out
                + torch.diag_embed(off_diagonal, offset=1)
                + torch.diag_embed(off_diagonal, offset=-1)
            )
        return out.contiguous()

    raise ValueError(f"unknown Cholesky test case: {case}")


def ref_kernel(data: input_t) -> output_t:
    return torch.linalg.cholesky_ex(data, check_errors=False).L


def _matrix_l1_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_norm(value, ord=1, dim=(-2, -1))


def check_implementation(data: input_t, output: output_t) -> tuple[bool, str]:
    if not isinstance(output, torch.Tensor):
        return False, "output must be a torch.Tensor"
    if output.shape != data.shape:
        return (
            False,
            f"output shape must be {tuple(data.shape)}, got {tuple(output.shape)}",
        )
    if output.dtype != torch.float32:
        return False, f"output dtype must be torch.float32, got {output.dtype}"
    if output.device != data.device:
        return False, f"output must be on {data.device}, got {output.device}"
    if not torch.isfinite(output).all().item():
        return False, "output contains NaN or Inf"

    batch, n, _ = data.shape
    eps = torch.finfo(torch.float32).eps
    scale = _matrix_l1_norm(data).clamp_min(torch.finfo(torch.float32).tiny)

    upper = torch.triu(output, diagonal=1)
    triangular_residual = _matrix_l1_norm(upper)
    triangular_allowed = _TRIANGULAR_RTOL_FACTOR * max(n, 1) * eps * scale
    if torch.any(triangular_residual > triangular_allowed).item():
        worst = (triangular_residual / scale).amax().item()
        return (
            False,
            f"output is not lower triangular enough: relative_residual={worst:.3g}",
        )

    diagonal = torch.diagonal(output, dim1=-2, dim2=-1)
    if torch.any(diagonal <= 0).item():
        return False, "output diagonal must be strictly positive"

    # Disable TF32 so the reconstruction gate measures FP32 factorization
    # accuracy rather than tensor-core matmul approximation error.
    old_tf32 = torch.backends.cuda.matmul.allow_tf32 if data.is_cuda else False
    try:
        if data.is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = False
        reconstruction = output @ output.transpose(-1, -2)
    finally:
        if data.is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    residual = _matrix_l1_norm(reconstruction - data)
    allowed = _RECON_RTOL_FACTOR * max(n, 1) * eps * scale
    if torch.any(residual > allowed).item():
        worst = (residual / scale).amax().item()
        return False, f"L @ L.T does not reconstruct A: relative_residual={worst:.3g}"

    scaled = (residual / (eps * max(n, 1) * scale)).amax().item()
    return True, f"scaled_reconstruction_residual={scaled:.3g}; batch={batch}; n={n}"
