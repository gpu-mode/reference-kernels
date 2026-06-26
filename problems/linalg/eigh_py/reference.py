import torch
from task import input_t, output_t


# Intentionally broad, dimension-scaled residual gates. Eigh has sign and
# eigenspace non-uniqueness, and we want to admit reasonable approximate or
# low-bit internal strategies without comparing against reference eigenvectors.
_EIGEN_RTOL_FACTOR = 200.0
_RECON_RTOL_FACTOR = 400.0
_ORTH_RTOL_FACTOR = 100.0
_SORT_RTOL_FACTOR = 100.0


def _matrix_l1_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_norm(value.double(), ord=1, dim=(-2, -1))


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


def _band_mask(n: int, bandwidth: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(n, device=device)
    return (idx[:, None] - idx[None, :]).abs() <= bandwidth


def _symmetrize(a: torch.Tensor) -> torch.Tensor:
    return 0.5 * (a + a.transpose(-1, -2))


def _signed_logspace(batch: int, n: int, cond: int, device: torch.device) -> torch.Tensor:
    span = max(cond, 1)
    magnitudes = torch.logspace(-float(span), 0.0, n, device=device, dtype=torch.float32)
    signs = torch.ones((n,), device=device, dtype=torch.float32)
    signs[::2] = -1.0
    values = magnitudes * signs
    return values.expand(batch, n).contiguous()


def _random_orthogonal(batch: int, n: int, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    x = torch.randn((batch, n, n), device=device, dtype=torch.float32, generator=gen)
    q, r = torch.linalg.qr(x)
    signs = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).clamp(min=0.0).mul(2.0).sub(1.0)
    return q * signs.unsqueeze(-2)


def _make_from_spectrum(values: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    batch, n = values.shape
    q = _random_orthogonal(batch, n, gen, values.device)
    a = (q * values.unsqueeze(-2)) @ q.transpose(-1, -2)
    return _symmetrize(a).contiguous()


def _apply_case(a: torch.Tensor, case: str, cond: int, gen: torch.Generator) -> torch.Tensor:
    batch, n, _ = a.shape
    device = a.device

    if case == "dense":
        a = _symmetrize(a)
        if cond:
            scales = torch.logspace(0.0, -float(cond), n, device=device, dtype=torch.float32)
            a = scales.reshape(1, n, 1) * a * scales.reshape(1, 1, n)
    elif case == "spectrum":
        values = _signed_logspace(batch, n, cond, device)
        a = _make_from_spectrum(values, gen)
    elif case == "psd":
        scales = torch.logspace(0.0, -float(max(cond, 1)), n, device=device, dtype=torch.float32)
        g = a * scales.reshape(1, 1, n)
        a = (g @ g.transpose(-1, -2)) / float(n)
    elif case == "rankdef":
        rank = max(1, (3 * n) // 4)
        values = torch.zeros((batch, n), device=device, dtype=torch.float32)
        values[:, -rank:] = torch.logspace(
            -float(max(cond, 1)), 0.0, rank, device=device, dtype=torch.float32
        )
        a = _make_from_spectrum(values, gen)
    elif case == "nearrank":
        rank = max(1, (3 * n) // 4)
        values = torch.empty((batch, n), device=device, dtype=torch.float32)
        values[:, : n - rank] = 1.0e-6 * torch.logspace(
            -2.0, 0.0, n - rank, device=device, dtype=torch.float32
        )
        values[:, n - rank :] = torch.logspace(
            -float(max(cond, 1)), 0.0, rank, device=device, dtype=torch.float32
        )
        a = _make_from_spectrum(values, gen)
    elif case == "repeated":
        groups = max(1, min(16, n // 8))
        base = torch.linspace(-1.0, 1.0, groups, device=device, dtype=torch.float32)
        values = base.repeat_interleave((n + groups - 1) // groups)[:n]
        values = values.expand(batch, n).contiguous()
        a = _make_from_spectrum(values, gen)
    elif case == "clustered":
        center = torch.linspace(-1.0, 1.0, n, device=device, dtype=torch.float32)
        jitter = torch.linspace(-1.0, 1.0, n, device=device, dtype=torch.float32)
        values = center.sign().clamp(min=0.0).mul(2.0).sub(1.0) + 1.0e-5 * jitter
        values[n // 3 : 2 * n // 3] = 1.0 + 1.0e-6 * jitter[n // 3 : 2 * n // 3]
        values = values.sort().values.expand(batch, n).contiguous()
        a = _make_from_spectrum(values, gen)
    elif case == "diagonal":
        values = _signed_logspace(batch, n, cond, device)
        a = torch.diag_embed(values)
    elif case == "band":
        bandwidth = max(2, min(32, n // 32))
        a = _symmetrize(a) * _band_mask(n, bandwidth, device)
        diag_boost = torch.linspace(-1.0, 1.0, n, device=device, dtype=torch.float32)
        a.diagonal(dim1=-2, dim2=-1).add_(diag_boost)
    elif case == "rowscale":
        row_cond = max(cond, 4)
        scales = torch.logspace(0.0, -float(row_cond), n, device=device, dtype=torch.float32)
        a = scales.reshape(1, n, 1) * _symmetrize(a) * scales.reshape(1, 1, n)
    else:
        raise ValueError(f"unknown eigh test case: {case}")

    return _symmetrize(a).contiguous()


_MIXED_PROFILES = (
    "dense",
    "spectrum",
    "psd",
    "rankdef",
    "nearrank",
    "repeated",
    "clustered",
    "band",
    "rowscale",
)
_MIXED_WEIGHTS = (6.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def _generate_mixed(a: torch.Tensor, cond: int, gen: torch.Generator) -> torch.Tensor:
    batch = a.shape[0]
    device = a.device
    weights = torch.tensor(_MIXED_WEIGHTS, dtype=torch.float32, device=device)
    labels = torch.multinomial(weights, batch, replacement=True, generator=gen)

    if batch >= 2:
        is_dense = labels == 0
        if not bool(is_dense.any()):
            labels[int(torch.randint(0, batch, (1,), device=device, generator=gen))] = 0
        elif bool(is_dense.all()):
            pos = int(torch.randint(0, batch, (1,), device=device, generator=gen))
            labels[pos] = int(torch.randint(1, len(_MIXED_PROFILES), (1,), device=device, generator=gen))

    for k, prof in enumerate(_MIXED_PROFILES):
        mask = labels == k
        if bool(mask.any()):
            a[mask] = _apply_case(a[mask], prof, cond, gen)
    return a


def generate_input(batch: int, n: int, cond: int, seed: int, case: str = "dense") -> input_t:
    assert batch > 0, "batch must be positive"
    assert n > 0, "n must be positive"
    assert cond >= 0, "cond must be non-negative"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    case = case.lower()
    a = torch.randn((batch, n, n), device=device, dtype=torch.float32, generator=gen)
    if case == "mixed":
        return _generate_mixed(a, cond, gen).contiguous()
    return _apply_case(a, case, cond, gen).contiguous()


def ref_kernel(data: input_t) -> output_t:
    values, vectors = torch.linalg.eigh(data)
    return vectors, values


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


def _check_ascending(values: torch.Tensor, n: int) -> tuple[bool, str]:
    if values.shape[-1] <= 1:
        return True, ""
    diffs = values[..., 1:] - values[..., :-1]
    scale = values.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    allowed = _property_rtol(n, _SORT_RTOL_FACTOR) * scale
    failed = diffs < -allowed
    if bool(failed.any().item()):
        matrix, col = torch.nonzero(failed, as_tuple=False)[0].tolist()
        return False, (
            "eigenvalues must be sorted in ascending order: "
            f"matrix={matrix}, index={col}, "
            f"left={values[matrix, col].item():.3g}, right={values[matrix, col + 1].item():.3g}"
        )
    return True, ""


def check_implementation(data: input_t, output: output_t) -> tuple[bool, str]:
    a = data
    batch, n, _ = a.shape
    eigen_rtol = _property_rtol(n, _EIGEN_RTOL_FACTOR)
    recon_rtol = _property_rtol(n, _RECON_RTOL_FACTOR)
    orth_rtol = _property_rtol(n, _ORTH_RTOL_FACTOR)

    if not isinstance(output, tuple) or len(output) != 2:
        return False, "output must be a tuple `(Q, L)`"

    q, values = output
    error = _check_tensor("Q", q, (batch, n, n), a.device)
    if error is not None:
        return False, error
    error = _check_tensor("L", values, (batch, n), a.device)
    if error is not None:
        return False, error

    good, message = _check_ascending(values, n)
    if not good:
        return False, message

    a_check = a.double()
    q_check = q.double()
    values_check = values.double()
    aq = a_check @ q_check
    ql = q_check * values_check.unsqueeze(-2)
    if not torch.isfinite(aq).all().item() or not torch.isfinite(ql).all().item():
        return False, "A @ Q or Q @ diag(L) contains NaN or Inf"

    eigen_residual = _matrix_l1_norm(aq - ql)
    eigen_scale = _matrix_l1_norm(a_check)
    eigen_allowed = eigen_rtol * eigen_scale
    eigen_scaled = _scaled_residual(eigen_residual, eigen_scale, n)
    if not torch.isfinite(eigen_scaled).all().item():
        return False, "A @ Q - Q @ diag(L) residual produced NaN or Inf"
    eigen_failed = eigen_residual > eigen_allowed
    if bool(eigen_failed.any().item()):
        worst = int(eigen_scaled.argmax().item())
        return False, (
            "A @ Q - Q @ diag(L) is too large: "
            f"matrix={worst}, residual={eigen_residual[worst].item():.3g}, "
            f"allowed={eigen_allowed[worst].item():.3g}, "
            f"scaled={eigen_scaled[worst].item():.3g}"
        )

    eye = torch.eye(n, device=a.device, dtype=torch.float64).expand(batch, n, n)
    qtq = q_check.transpose(-1, -2) @ q_check
    if not torch.isfinite(qtq).all().item():
        return False, "Q.T @ Q contains NaN or Inf"
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

    recon = ql @ q_check.transpose(-1, -2)
    if not torch.isfinite(recon).all().item():
        return False, "Q @ diag(L) @ Q.T contains NaN or Inf"
    recon_residual = _matrix_l1_norm(recon - a_check)
    recon_scale = _matrix_l1_norm(a_check)
    recon_allowed = recon_rtol * recon_scale
    recon_scaled = _scaled_residual(recon_residual, recon_scale, n)
    recon_failed = recon_residual > recon_allowed
    if bool(recon_failed.any().item()):
        worst = int(recon_scaled.argmax().item())
        return False, (
            "Q @ diag(L) @ Q.T reconstruction is too large: "
            f"matrix={worst}, residual={recon_residual[worst].item():.3g}, "
            f"allowed={recon_allowed[worst].item():.3g}, "
            f"scaled={recon_scaled[worst].item():.3g}"
        )

    projected = q_check.transpose(-1, -2) @ a_check @ q_check
    offdiag = projected - torch.diag_embed(torch.diagonal(projected, dim1=-2, dim2=-1))
    diag_residual = _matrix_l1_norm(offdiag).amax()
    diag_scale = _matrix_l1_norm(a_check).amax()
    diag_scaled = _scaled_residual(diag_residual, diag_scale, n)

    return True, (
        f"eigen_rtol={eigen_rtol:.3g}; "
        f"recon_rtol={recon_rtol:.3g}; "
        f"orth_rtol={orth_rtol:.3g}; "
        f"scaled_eigen_residual={eigen_scaled.amax().item():.3g}; "
        f"scaled_reconstruction_residual={recon_scaled.amax().item():.3g}; "
        f"scaled_diagonalization_residual={diag_scaled.item():.3g}; "
        f"scaled_orthogonality_residual={orth_scaled.item():.3g}; "
        f"batch={batch}; n={n}"
    )
