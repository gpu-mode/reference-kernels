import torch
from task import input_t, output_t
from utils import verbose_allclose

CHUNK_SIZE = 64


def generate_input(B: int, T: int, H: int, K: int, V: int, seed: int) -> input_t:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    k = torch.randn(B, T, H, K, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    w = torch.randn(B, T, H, K, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    u = torch.randn(B, T, H, V, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    # Use negative values for g to keep exp(g) bounded in (0, 1] and prevent overflow
    g = -torch.abs(torch.randn(B, T, H, dtype=torch.float32, device="cuda", generator=gen)).contiguous()
    return k, w, u, g


def ref_kernel(data: input_t) -> output_t:
    k, w, u, g = data
    B, T, H, K = k.shape
    V = u.shape[-1]
    BT = CHUNK_SIZE
    NT = T // BT

    h = torch.empty(B, NT, H, K, V, dtype=torch.float32, device=k.device)
    v_new = torch.empty_like(u)

    for b in range(B):
        for hh in range(H):
            b_h = torch.zeros(K, V, dtype=torch.float32, device=k.device)

            for c in range(NT):
                cs = c * BT
                ce = cs + BT

                # Store current state
                h[b, c, hh] = b_h

                # v_new = u - w @ h_state
                b_w = w[b, cs:ce, hh].float()  # [BT, K]
                b_u = u[b, cs:ce, hh].float()  # [BT, V]
                b_v = b_u - torch.matmul(b_w, b_h)  # [BT, V]
                v_new[b, cs:ce, hh] = b_v

                # Gating
                b_g = g[b, cs:ce, hh].float()  # [BT]
                b_g_last = b_g[-1]
                b_v_gated = b_v * torch.exp(b_g_last - b_g)[:, None]

                # Decay and update
                b_h = b_h * torch.exp(b_g_last)
                b_k = k[b, cs:ce, hh].float()  # [BT, K]
                b_h = b_h + torch.matmul(b_k.T, b_v_gated)

    return h, v_new


def check_implementation(data, output):
    expected = ref_kernel(data)
    exp_h, exp_v = expected
    got_h, got_v = output

    reasons_h = verbose_allclose(got_h, exp_h, rtol=2e-2, atol=2e-2)
    reasons_v = verbose_allclose(got_v, exp_v, rtol=2e-2, atol=2e-2)

    reasons = []
    if reasons_h:
        reasons.append("h mismatch: " + " ".join(reasons_h))
    if reasons_v:
        reasons.append("v_new mismatch: " + " ".join(reasons_v))

    if reasons:
        return False, " | ".join(reasons)
    return True, ""
