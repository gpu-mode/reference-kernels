import torch
from task import input_t, output_t
from utils import verbose_allclose

CHUNK_SIZE = 64


def generate_input(B: int, T: int, H: int, K: int, V: int, seed: int) -> input_t:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    k = torch.randn(B, T, H, K, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    v = torch.randn(B, T, H, V, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    beta = torch.randn(B, T, H, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    A = torch.randn(B, T, H, CHUNK_SIZE, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    # Use negative values for g to keep exp(g) bounded in (0, 1]
    g = -torch.abs(torch.randn(B, T, H, dtype=torch.float32, device="cuda", generator=gen)).contiguous()
    return k, v, beta, A, g


def ref_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = CHUNK_SIZE

    w = torch.empty_like(k)
    u = torch.empty_like(v)

    for cs in range(0, T, BT):
        ce = cs + BT
        # Reshape to [B, H, BT, BT] for batched matmul
        A_bh = A[:, cs:ce, :, :].permute(0, 2, 1, 3).float()

        # u = A @ (v * beta[..., None])
        vb = (v[:, cs:ce, :, :] * beta[:, cs:ce, :, None]).permute(0, 2, 1, 3).float()
        u[:, cs:ce, :, :] = torch.matmul(A_bh, vb).permute(0, 2, 1, 3)

        # w = A @ (k * beta[..., None] * exp(g)[..., None])
        kb = (k[:, cs:ce, :, :] * beta[:, cs:ce, :, None] * torch.exp(g[:, cs:ce, :, None])).permute(0, 2, 1, 3).float()
        w[:, cs:ce, :, :] = torch.matmul(A_bh, kb).permute(0, 2, 1, 3)

    return w, u


def check_implementation(data, output):
    expected = ref_kernel(data)
    exp_w, exp_u = expected
    got_w, got_u = output

    reasons_w = verbose_allclose(got_w, exp_w, rtol=1e-3, atol=1e-3)
    reasons_u = verbose_allclose(got_u, exp_u, rtol=1e-3, atol=1e-3)

    reasons = []
    if reasons_w:
        reasons.append("w mismatch: " + " ".join(reasons_w))
    if reasons_u:
        reasons.append("u mismatch: " + " ".join(reasons_u))

    if reasons:
        return False, " | ".join(reasons)
    return True, ""
