import torch
from task import input_t, output_t
from utils import make_match_reference

CHUNK_SIZE = 64


def generate_input(B: int, T: int, H: int, K: int, V: int, seed: int) -> input_t:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    NT = T // CHUNK_SIZE
    q = torch.randn(B, T, H, K, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    k = torch.randn(B, T, H, K, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    v_new = torch.randn(B, T, H, V, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    h = torch.randn(B, NT, H, K, V, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    # Use negative values for g to keep exp(g) bounded in (0, 1]
    g = -torch.abs(torch.randn(B, T, H, dtype=torch.float32, device="cuda", generator=gen)).contiguous()
    return q, k, v_new, h, g


def ref_kernel(data: input_t) -> output_t:
    q, k, v_new, h, g = data
    B, T, H, K = q.shape
    V = v_new.shape[-1]
    BT = CHUNK_SIZE
    scale = K ** -0.5

    o = torch.empty_like(v_new)
    causal = torch.tril(torch.ones(BT, BT, device=q.device, dtype=torch.bool))

    for cs in range(0, T, BT):
        ce = cs + BT
        c_idx = cs // BT

        # Reshape to [B, H, BT, ...] for batched matmul
        b_q = q[:, cs:ce, :, :].permute(0, 2, 1, 3).float()   # [B, H, BT, K]
        b_k = k[:, cs:ce, :, :].permute(0, 2, 1, 3).float()   # [B, H, BT, K]
        b_v = v_new[:, cs:ce, :, :].permute(0, 2, 1, 3).float()  # [B, H, BT, V]
        b_h = h[:, c_idx, :, :, :].float()                     # [B, H, K, V]
        b_g = g[:, cs:ce, :].permute(0, 2, 1).float()          # [B, H, BT]

        # Inter-chunk: q @ h * exp(g)
        inter = torch.matmul(b_q, b_h)  # [B, H, BT, V]
        inter = inter * torch.exp(b_g).unsqueeze(-1)

        # Intra-chunk: causal(q @ k^T * exp(g_diff)) @ v_new
        attn = torch.matmul(b_q, b_k.transpose(-1, -2))  # [B, H, BT, BT]
        g_diff = b_g.unsqueeze(-1) - b_g.unsqueeze(-2)    # [B, H, BT, BT]
        attn = attn * torch.exp(g_diff)
        attn = attn.masked_fill(~causal, 0.0)
        intra = torch.matmul(attn, b_v)  # [B, H, BT, V]

        b_o = (inter + intra) * scale
        o[:, cs:ce, :, :] = b_o.permute(0, 2, 1, 3)

    return o


check_implementation = make_match_reference(ref_kernel, rtol=1e-3, atol=1e-3)
