from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Per-shape configs: map (B, T, H, K, V) to optimized helion.Config objects.
# Autotune locally for each shape, then paste the best config here.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 2, 64, 64): helion.Config(...),  # TODO: replace with default config or any config that passes correctness check
    (2, 128, 4, 64, 64): helion.Config(...),  # TODO: replace with default config or any config that passes correctness check
    (1, 256, 4, 64, 128): helion.Config(...),  # TODO: replace with default config or any config that passes correctness check
    # Benchmark shapes
    (1, 64, 1, 64, 64): helion.Config(...),  # TODO: replace with your autotuned config
    (2, 512, 3, 64, 64): helion.Config(...),  # TODO: replace with your autotuned config
    (2, 1024, 3, 64, 64): helion.Config(...),  # TODO: replace with your autotuned config
    (3, 1024, 4, 100, 100): helion.Config(...),  # TODO: replace with your autotuned config
    (4, 1024, 4, 128, 128): helion.Config(...),  # TODO: replace with your autotuned config
    (2, 1536, 4, 128, 128): helion.Config(...),  # TODO: replace with your autotuned config
    (4, 2048, 8, 64, 64): helion.Config(...),  # TODO: replace with your autotuned config
}


# NOTE: This is an intentionally inefficient baseline implementation.
def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
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

    return project_kv


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    kernel = _KERNELS[(B, T, H, K, V)]
    return kernel(k, v, beta, A, g)
