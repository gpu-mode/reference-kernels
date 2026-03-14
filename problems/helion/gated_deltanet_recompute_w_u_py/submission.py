from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Per-shape configs: map (B, T, H, K, V) to optimized helion.Config objects.
# Autotune locally for each shape, then paste the best config here.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=3),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=3),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[64, 128], num_warps=4, num_stages=3),
    # Benchmark shapes
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=3),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=3),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=3),
    (3, 1024, 4, 100, 100): helion.Config(block_sizes=[100, 100], num_warps=4, num_stages=3),
    (4, 1024, 4, 128, 128): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=3),
    (2, 1536, 4, 128, 128): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=3),
    (4, 2048, 8, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=3),
}


# Optional: add advanced_controls_file to your Config for extra performance (see docs).
# Autotune with autotune_search_acf to find the best ACF, then hardcode it:
#     helion.Config(..., advanced_controls_file="/opt/booster_pack/recompute_w_u_fwd_0.acf")


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
    def kernel(
        k: torch.Tensor,     # [B, T, H, K]
        v: torch.Tensor,     # [B, T, H, V]
        beta: torch.Tensor,  # [B, T, H]
        A: torch.Tensor,     # [B, T, H, C]
        g: torch.Tensor,     # [B, T, H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H, K = k.shape
        V = v.shape[-1]
        C = hl.specialize(A.shape[-1])
        K = hl.specialize(K)
        V = hl.specialize(V)

        w_out = torch.empty_like(k)
        u_out = torch.empty_like(v)

        block_k = hl.register_block_size(K)
        block_v = hl.register_block_size(V)

        # W = A_mat @ (β * exp(g̃) * k)
        for tile_b, tile_h, rt, tile_k in hl.tile([B, H, T, K], block_size=[1, 1, C, block_k]):
            b = tile_b.begin
            h = tile_h.begin
            a_mat_w = A[b, rt, h, :].to(torch.float32)                        # [C, C]
            coeff_w = (beta[b, rt, h] * torch.exp(g[b, rt, h])).to(torch.float32)  # [C]
            sk = k[b, rt, h, tile_k].to(torch.float32) * coeff_w[:, None]    # [C, block_k]
            w_out[b, rt, h, tile_k] = hl.dot(a_mat_w, sk, out_dtype=torch.float32).to(k.dtype)

        # U = A_mat @ (β * v)
        for tile_b2, tile_h2, rt2, tile_v in hl.tile([B, H, T, V], block_size=[1, 1, C, block_v]):
            b2 = tile_b2.begin
            h2 = tile_h2.begin
            a_mat_u = A[b2, rt2, h2, :].to(torch.float32)                    # [C, C]
            bv = beta[b2, rt2, h2].to(torch.float32)                         # [C]
            sv = v[b2, rt2, h2, tile_v].to(torch.float32) * bv[:, None]      # [C, block_v]
            u_out[b2, rt2, h2, tile_v] = hl.dot(a_mat_u, sv, out_dtype=torch.float32).to(v.dtype)

        return w_out, u_out

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    kernel = _KERNELS[(B, T, H, K, V)]
    return kernel(k, v, beta, A, g)
