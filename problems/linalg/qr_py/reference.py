import torch
from task import input_t, output_t


_FACTOR_RTOL_FACTOR = 20.0
_ORTH_RTOL_FACTOR = 100.0


def _apply_column_scaling(a: torch.Tensor, cond: int) -> torch.Tensor:
    # `cond` is a deterministic dynamic-range knob, not an exact condition number.
    if cond:
        n = a.shape[-1]
        scales = torch.logspace(0.0, -float(cond), n, device=a.device, dtype=torch.float32)
        return a * scales
    return a.contiguous()


def _band_mask(n: int, bandwidth: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(n, device=device)
    return (idx[:, None] - idx[None, :]).abs() <= bandwidth


def generate_input(batch: int, n: int, cond: int, seed: int, case: str = "dense") -> input_t:
    assert batch > 0, "batch must be positive"
    assert n > 0, "n must be positive"
    assert cond >= 0, "cond must be non-negative"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    case = case.lower()
    a = torch.randn((batch, n, n), device=device, dtype=torch.float32, generator=gen)

    if case == "dense":
        a = _apply_column_scaling(a, cond)
    elif case == "upper":
        diag_boost = torch.linspace(1.0, 0.25, n, device=device, dtype=torch.float32)
        a = torch.triu(a)
        a.diagonal(dim1=-2, dim2=-1).add_(diag_boost)
        a = _apply_column_scaling(a, cond)
    elif case == "diagonal":
        diag = torch.randn((batch, n), device=device, dtype=torch.float32, generator=gen)
        diag = diag.sign().clamp(min=0.0).mul(2.0).sub(1.0) * torch.logspace(
            0.0, -float(max(cond, 2)), n, device=device, dtype=torch.float32
        )
        a = torch.diag_embed(diag)
    elif case == "rankdef":
        rank = max(1, (3 * n) // 4)
        a[:, :, rank:] = 0.0
        a = _apply_column_scaling(a, cond)
    elif case == "nearrank":
        rank = max(1, (3 * n) // 4)
        tail = n - rank
        if tail > 0:
            noise = torch.randn(
                (batch, n, tail), device=device, dtype=torch.float32, generator=gen
            )
            a[:, :, rank:] = a[:, :, :tail] + 1.0e-5 * noise
        a = _apply_column_scaling(a, cond)
    elif case == "clustered":
        scales = torch.ones((n,), device=device, dtype=torch.float32)
        scales[n // 2 :] = 4.0 * torch.finfo(torch.float32).eps
        if n >= 8:
            lo = max(0, n // 2 - 2)
            hi = min(n, n // 2 + 2)
            scales[lo:hi] = torch.sqrt(torch.tensor(torch.finfo(torch.float32).eps, device=device))
        a = a * scales
    elif case == "band":
        bandwidth = max(2, min(32, n // 32))
        a = a * _band_mask(n, bandwidth, device)
        diag_boost = torch.linspace(1.0, 0.5, n, device=device, dtype=torch.float32)
        a.diagonal(dim1=-2, dim2=-1).add_(diag_boost)
        a = _apply_column_scaling(a, cond)
    elif case == "nearcollinear":
        base = torch.randn((batch, n, 1), device=device, dtype=torch.float32, generator=gen)
        noise = torch.randn((batch, n, n), device=device, dtype=torch.float32, generator=gen)
        a = base.expand(batch, n, n) + 1.0e-4 * noise
        a = _apply_column_scaling(a, cond)
    elif case == "rowscale":
        row_cond = max(cond, 4)
        scales = torch.logspace(0.0, -float(row_cond), n, device=device, dtype=torch.float32)
        a = scales.reshape(1, n, 1) * a
    else:
        raise ValueError(f"unknown QR test case: {case}")

    return a.contiguous()


def ref_kernel(data: input_t) -> output_t:
    # Starter/reference path: correctness first; submissions compete on speed.
    return torch.geqrf(data)


def _property_rtol(n: int, factor: float) -> float:
    eps = torch.finfo(torch.float32).eps
    return factor * max(n, 1) * eps


def _scaled_residual(
    residual: torch.Tensor,
    scale: torch.Tensor,
    n: int,
) -> torch.Tensor:
    eps = torch.finfo(torch.float32).eps
    return residual / (eps * max(n, 1) * scale.clamp_min(1e-30))


def _matrix_l1_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_norm(value.double(), ord=1, dim=(-2, -1))


def _check_tensor(name: str, value: torch.Tensor, shape: tuple[int, ...], device: torch.device) -> str | None:
    if not isinstance(value, torch.Tensor):
        return f"{name} must be a torch.Tensor"
    if value.shape != shape:
        return f"{name} shape must be {shape}, got {tuple(value.shape)}"
    if value.dtype != torch.float32:
        return f"{name} dtype must be torch.float32, got {value.dtype}"
    if value.device != device:
        return f"{name} must be on {device}, got {value.device}"
    if not torch.isfinite(value).all().item():
        return f"{name} contains NaN or Inf"
    return None


def check_implementation(data: input_t, output: output_t) -> tuple[bool, str]:
    a = data
    batch, n, _ = a.shape
    factor_rtol = _property_rtol(n, _FACTOR_RTOL_FACTOR)
    orth_rtol = _property_rtol(n, _ORTH_RTOL_FACTOR)

    if not isinstance(output, tuple) or len(output) != 2:
        return False, "output must be a tuple `(H, tau)`"

    h, tau = output
    error = _check_tensor("H", h, (batch, n, n), a.device)
    if error is not None:
        return False, error
    error = _check_tensor("tau", tau, (batch, n), a.device)
    if error is not None:
        return False, error

    q = torch.linalg.householder_product(h, tau)
    r = torch.triu(h)
    a_check = a.double()
    q_check = q.double()
    r_check = r.double()
    projected = q_check.transpose(-1, -2) @ a_check
    factor_residual = _matrix_l1_norm(r_check - projected).amax()
    factor_scale = _matrix_l1_norm(a_check).amax()
    factor_allowed = factor_rtol * factor_scale
    factor_scaled = _scaled_residual(factor_residual, factor_scale, n)
    if factor_residual.item() > factor_allowed.item():
        return False, (
            "R - Q.T @ A is too large: "
            f"residual={factor_residual.item():.3g}, allowed={factor_allowed.item():.3g}, "
            f"scaled={factor_scaled.item():.3g}"
        )

    eye = torch.eye(n, device=a.device, dtype=torch.float64).expand(batch, n, n)
    qtq = q_check.transpose(-1, -2) @ q_check
    orth_residual = _matrix_l1_norm(qtq - eye).amax()
    orth_scale = _matrix_l1_norm(eye).amax()
    orth_allowed = orth_rtol * orth_scale
    orth_scaled = _scaled_residual(orth_residual, orth_scale, n)
    if orth_residual.item() > orth_allowed.item():
        return False, (
            "Q is not orthogonal enough: "
            f"residual={orth_residual.item():.3g}, allowed={orth_allowed.item():.3g}, "
            f"scaled={orth_scaled.item():.3g}"
        )

    lower = torch.tril(projected, diagonal=-1)
    tri_residual = _matrix_l1_norm(lower).amax()
    tri_scale = _matrix_l1_norm(a_check).amax()
    tri_scaled = _scaled_residual(tri_residual, tri_scale, n)

    recon = q_check @ r_check
    recon_residual = _matrix_l1_norm(recon - a_check).amax()
    recon_scale = _matrix_l1_norm(a_check).amax()
    recon_scaled = _scaled_residual(recon_residual, recon_scale, n)

    return True, (
        f"factor_rtol={factor_rtol:.3g}; "
        f"orth_rtol={orth_rtol:.3g}; "
        f"scaled_factor_residual={factor_scaled.item():.3g}; "
        f"scaled_reconstruction_residual={recon_scaled.item():.3g}; "
        f"scaled_triangular_residual={tri_scaled.item():.3g}; "
        f"scaled_orthogonality_residual={orth_scaled.item():.3g}; "
        f"batch={batch}; n={n}"
    )
