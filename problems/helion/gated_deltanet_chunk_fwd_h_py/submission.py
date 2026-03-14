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
#     helion.Config(..., advanced_controls_file="/opt/booster_pack/chunk_fwd_h_0.acf")


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
    def kernel(
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
        block_v = hl.register_block_size(V)

        for tile_b, tile_h, tile_v in hl.tile([B, H, V], block_size=[1, 1, block_v]):
            b_idx = tile_b.id
            h_idx = tile_h.id
            state = hl.zeros([K, tile_v], dtype=torch.float32)

            for tc in hl.tile(T, block_size=C):
                t_end = min(tc.begin + C, T) - 1

                h_out[b_idx, tc.id, h_idx, :, tile_v] = state.to(k.dtype)

                b_w = w[b_idx, tc, h_idx, :]
                proj = hl.dot(b_w, state.to(k.dtype), out_dtype=torch.float32)
                p_v = u[b_idx, tc, h_idx, tile_v].to(torch.float32)
                diff = p_v - proj
                v_out[b_idx, tc, h_idx, tile_v] = diff.to(u.dtype)

                g_end = g[b_idx, t_end, h_idx].to(torch.float32)
                b_g = g[b_idx, tc, h_idx].to(torch.float32)
                valid = tc.index < T
                alpha = torch.where(valid, torch.exp(g_end - b_g), 0.0)
                diff_gated = (diff * alpha[:, None]).to(k.dtype)

                state = state * torch.exp(g_end)
                b_k = k[b_idx, tc, h_idx, :]
                state = hl.dot(b_k.T, diff_gated, acc=state)

        return h_out, v_out

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    k, w, u, g = data
    B, T, H, K = k.shape
    V = u.shape[-1]
    kernel = _KERNELS[(B, T, H, K, V)]
    return kernel(k, w, u, g)
