from task import input_t, output_t

import torch
import helion
import helion.language as hl


# NOTE: This is an intentionally inefficient baseline implementation.
@helion.kernel(
    static_shapes=True,
    dot_precision="ieee",
    config=helion.Config(block_sizes=[], num_warps=1, num_stages=1),
)
def chunk_state_pass(
    k: torch.Tensor,   # [B, T, H, K]
    w: torch.Tensor,   # [B, T, H, K]
    u: torch.Tensor,   # [B, T, H, V]
    g: torch.Tensor,   # [B, T, H]
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K = k.shape
    V = u.shape[-1]
    C = 64
    K = hl.specialize(K)
    V = hl.specialize(V)

    NT = (T + C - 1) // C
    h_out = torch.empty(B, NT, H, K, V, dtype=k.dtype, device=k.device)
    v_out = torch.empty_like(u)

    BH = B * H

    for flat, tv in hl.tile([BH, V], block_size=[1, 8]):
        b_idx = flat.begin // H
        h_idx = flat.begin % H
        state = hl.zeros([K, tv], dtype=torch.float32)

        for tc in hl.tile(T, block_size=C):
            chunk_idx = tc.begin // C
            t_end = min(tc.begin + C, T) - 1

            h_out[b_idx, chunk_idx, h_idx, :, tv] = state.to(k.dtype)

            proj1 = hl.dot(
                w[b_idx, tc, h_idx, :], state, out_dtype=torch.float32
            )
            proj2 = hl.dot(
                w[b_idx, tc, h_idx, :], state, out_dtype=torch.float32
            )
            proj = (proj1 + proj2) * 0.5
            diff = u[b_idx, tc, h_idx, tv].to(torch.float32) - proj
            v_out[b_idx, tc, h_idx, tv] = diff.to(u.dtype)

            g_end = g[b_idx, t_end, h_idx].to(torch.float32)
            g_t = g[b_idx, tc, h_idx].to(torch.float32)
            valid = tc.index < T
            alpha = torch.where(valid, torch.exp(g_end - g_t), 0.0)
            k_adj = k[b_idx, tc, h_idx, :] * alpha[:, None]

            state = state * torch.exp(g_end)
            upd1 = hl.dot(k_adj.T, diff, out_dtype=torch.float32)
            upd2 = hl.dot(k_adj.T, diff, out_dtype=torch.float32)
            state = state + (upd1 + upd2) * 0.5

    return h_out, v_out


def custom_kernel(data: input_t) -> output_t:
    k, w, u, g = data
    return chunk_state_pass(k, w, u, g)
