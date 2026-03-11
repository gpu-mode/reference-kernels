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
def project_kv(
    k: torch.Tensor,     # [B, T, H, K]
    v: torch.Tensor,     # [B, T, H, V]
    beta: torch.Tensor,  # [B, T, H]
    A: torch.Tensor,     # [B, T, H, BT]
    g: torch.Tensor,     # [B, T, H]
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K = k.shape
    V = v.shape[-1]
    C = hl.specialize(A.shape[-1])
    K = hl.specialize(K)
    V = hl.specialize(V)

    w_out = torch.empty_like(k)
    u_out = torch.empty_like(v)

    BH = B * H
    for flat_bh, rt in hl.tile([BH, T], block_size=[1, C]):
        b_idx = flat_bh.begin // H
        h_idx = flat_bh.begin % H

        w_acc1 = hl.zeros([rt, K], dtype=torch.float32)
        u_acc1 = hl.zeros([rt, V], dtype=torch.float32)
        w_acc2 = hl.zeros([rt, K], dtype=torch.float32)
        u_acc2 = hl.zeros([rt, V], dtype=torch.float32)

        for ci in range(C):
            t_ci = rt.begin + ci
            a_col = A[b_idx, rt, h_idx, ci].to(torch.float32)
            coeff_ci = beta[b_idx, t_ci, h_idx].to(torch.float32)
            decay_ci = torch.exp(g[b_idx, t_ci, h_idx].to(torch.float32))

            k_ci = k[b_idx, t_ci, h_idx, :].to(torch.float32)
            v_ci = v[b_idx, t_ci, h_idx, :].to(torch.float32)

            w_acc1 = w_acc1 + a_col[:, None] * (k_ci * coeff_ci * decay_ci)[None, :]
            u_acc1 = u_acc1 + a_col[:, None] * (v_ci * coeff_ci)[None, :]

        for ci in range(C - 1, -1, -1):
            t_ci = rt.begin + ci
            a_col = A[b_idx, rt, h_idx, ci].to(torch.float32)
            coeff_ci = beta[b_idx, t_ci, h_idx].to(torch.float32)
            decay_ci = torch.exp(g[b_idx, t_ci, h_idx].to(torch.float32))

            k_ci = k[b_idx, t_ci, h_idx, :].to(torch.float32)
            v_ci = v[b_idx, t_ci, h_idx, :].to(torch.float32)

            w_acc2 = w_acc2 + a_col[:, None] * (k_ci * coeff_ci * decay_ci)[None, :]
            u_acc2 = u_acc2 + a_col[:, None] * (v_ci * coeff_ci)[None, :]

        w_out[b_idx, rt, h_idx, :] = ((w_acc1 + w_acc2) * 0.5).to(k.dtype)
        u_out[b_idx, rt, h_idx, :] = ((u_acc1 + u_acc2) * 0.5).to(v.dtype)

    return w_out, u_out


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    return project_kv(k, v, beta, A, g)
