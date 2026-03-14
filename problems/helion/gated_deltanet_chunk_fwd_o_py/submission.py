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


# Optional: add advanced_controls_file to your Config for extra performance (see docs).
# Autotune with autotune_search_acf to find the best ACF, then hardcode it:
#     helion.Config(..., advanced_controls_file="/opt/booster_pack/chunk_fwd_o_0.acf")


# NOTE: This is an intentionally inefficient baseline implementation.
def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
    def kernel(
        q: torch.Tensor,     # [B, T, H, K]
        k: torch.Tensor,     # [B, T, H, K]
        v: torch.Tensor,     # [B, T, H, V]
        h: torch.Tensor,     # [B, NT, H, K, V]
        g: torch.Tensor,     # [B, T, H]
        scale: float,
    ) -> torch.Tensor:
        B, T, H, K = q.shape
        V = v.shape[-1]
        C = 64
        K = hl.specialize(K)
        V = hl.specialize(V)

        out = torch.empty_like(v)

        BH = B * H
        for flat_bh, tile_t in hl.tile([BH, T], block_size=[1, C]):
            b_idx = flat_bh.begin // H
            h_idx = flat_bh.begin % H
            c_idx = tile_t.begin // C

            g_vals = g[b_idx, tile_t, h_idx]
            q_s = q[b_idx, tile_t, h_idx, :] * torch.exp(g_vals)[:, None]
            k_s = k[b_idx, tile_t, h_idx, :] * torch.exp(-g_vals)[:, None]

            sim1 = hl.dot(q_s, k_s.T)
            sim2 = hl.dot(q_s, k_s.T)
            sim = (sim1 + sim2) * 0.5
            idx = hl.arange(tile_t.block_size)
            mask = idx[:, None] >= idx[None, :]
            sim = torch.where(mask, sim, 0.0)
            local1 = hl.dot(sim.to(v.dtype), v[b_idx, tile_t, h_idx, :])
            local2 = hl.dot(sim.to(v.dtype), v[b_idx, tile_t, h_idx, :])
            local_out = (local1 + local2) * 0.5

            glob1 = hl.dot(q_s, h[b_idx, c_idx, h_idx, :, :])
            glob2 = hl.dot(q_s, h[b_idx, c_idx, h_idx, :, :])
            global_out = (glob1 + glob2) * 0.5

            out[b_idx, tile_t, h_idx, :] = ((global_out + local_out) * scale).to(out.dtype)

        return out

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    q, k, v_new, h, g = data
    B, T, H, K = q.shape
    V = v_new.shape[-1]
    scale = K ** -0.5
    kernel = _KERNELS[(B, T, H, K, V)]
    return kernel(q, k, v_new, h, g, scale)
