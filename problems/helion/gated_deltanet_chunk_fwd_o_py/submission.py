from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Per-shape configs: map (B, T, H, K, V) to optimized helion.Config objects.
# Autotune locally for each shape, then paste the best config here.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[64], num_warps=4, num_stages=2),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[64], num_warps=4, num_stages=2),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[128], num_warps=4, num_stages=2),
    # Benchmark shapes
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[64], num_warps=4, num_stages=2),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[64], num_warps=4, num_stages=2),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[64], num_warps=4, num_stages=2),
    (3, 1024, 4, 100, 100): helion.Config(block_sizes=[100], num_warps=4, num_stages=2),
    (4, 1024, 4, 128, 128): helion.Config(block_sizes=[128], num_warps=4, num_stages=2),
    (2, 1536, 4, 128, 128): helion.Config(block_sizes=[128], num_warps=4, num_stages=2),
    (4, 2048, 8, 64, 64): helion.Config(block_sizes=[64], num_warps=4, num_stages=2),
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

        block_v = hl.register_block_size(V)
        for tile_b, tile_h, tile_t, tile_v in hl.tile([B, H, T, V], block_size=[1, 1, C, block_v]):
            b_idx = tile_b.begin
            h_idx = tile_h.begin
            c_idx = tile_t.begin // C

            g_vals = g[b_idx, tile_t, h_idx]
            q_tile = q[b_idx, tile_t, h_idx, :]
            k_tile = k[b_idx, tile_t, h_idx, :]
            v_tile = v[b_idx, tile_t, h_idx, tile_v]

            # intra-chunk: q @ k^T * exp(g_i - g_j), with causal mask
            qk = hl.dot(q_tile, k_tile.T)
            idx = hl.arange(tile_t.block_size)
            g_diff = g_vals[:, None] - g_vals[None, :]
            causal_mask = idx[:, None] >= idx[None, :]
            sim = torch.where(causal_mask, qk * torch.exp(g_diff), 0.0)
            local_out = hl.dot(sim.to(v.dtype), v_tile)

            # inter-chunk: (q @ h) * exp(g)
            q_s = q_tile * torch.exp(g_vals)[:, None]
            global_out = hl.dot(q_s, h[b_idx, c_idx, h_idx, :, tile_v])

            out[b_idx, tile_t, h_idx, tile_v] = ((global_out + local_out) * scale).to(out.dtype)

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
