from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Per-shape configs: map (B, D, S, W) to optimized helion.Config objects.
# Autotune locally for each shape, then paste the best config here.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 64, 4): helion.Config(...),  # TODO: replace with default config or any config that passes correctness check
    (2, 128, 128, 4): helion.Config(...),  # TODO: replace with default config or any config that passes correctness check
    (1, 256, 256, 3): helion.Config(...),  # TODO: replace with default config or any config that passes correctness check
    (1, 128, 64, 8): helion.Config(...),  # TODO: replace with default config or any config that passes correctness check
    (4, 64, 128, 4): helion.Config(...),  # TODO: replace with default config or any config that passes correctness check
    # Benchmark shapes
    (1, 768, 512, 4): helion.Config(...),  # TODO: replace with your autotuned config
    (1, 768, 2048, 4): helion.Config(...),  # TODO: replace with your autotuned config
    (1, 1536, 2048, 4): helion.Config(...),  # TODO: replace with your autotuned config
    (1, 2560, 2048, 4): helion.Config(...),  # TODO: replace with your autotuned config
    (1, 2560, 4096, 4): helion.Config(...),  # TODO: replace with your autotuned config
}


# Optional: add advanced_controls_file to your Config for extra performance (see docs).
# Autotune with autotune_search_acf to find the best ACF, then hardcode it:
#     helion.Config(..., advanced_controls_file="/opt/booster_pack/causal_conv_0.acf")


# NOTE: This is an intentionally inefficient baseline implementation.
def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(
        x_pad: torch.Tensor,  # (B, D, L) zero-padded input
        w: torch.Tensor,      # (D, W) filter coefficients
        b: torch.Tensor,      # (D,) additive offset
    ) -> torch.Tensor:
        B = x_pad.size(0)
        D = x_pad.size(1)
        L = x_pad.size(2)
        W = hl.specialize(w.size(1))
        N = L - W + 1

        y = torch.empty(B, D, N, dtype=x_pad.dtype, device=x_pad.device)

        for rb, rd, rs in hl.tile([B, D, N], block_size=[1, None, None]):
            bi = rb.begin
            acc1 = hl.zeros([rd, rs], dtype=torch.float32)
            acc2 = hl.zeros([rd, rs], dtype=torch.float32)
            acc3 = hl.zeros([rd, rs], dtype=torch.float32)
            for j in range(W):
                c1 = w[rd, j].to(torch.float32)
                x1 = hl.load(x_pad, [bi, rd, rs.index + j]).to(torch.float32)
                acc1 = acc1 + x1 * c1[:, None]
                c2 = w[rd, j].to(torch.float32)
                x2 = hl.load(x_pad, [bi, rd, rs.index + j]).to(torch.float32)
                acc2 = acc2 + x2 * c2[:, None]
                c3 = w[rd, j].to(torch.float32)
                x3 = hl.load(x_pad, [bi, rd, rs.index + j]).to(torch.float32)
                acc3 = acc3 + x3 * c3[:, None]
            acc = (acc1 + acc2 + acc3) / 3.0
            acc = acc + b[rd].to(torch.float32)[:, None]
            y[rb, rd, rs] = acc[None, :, :].to(y.dtype)

        return y

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data
    B, D, S = x.shape
    W = weight.shape[1]
    kernel = _KERNELS[(B, D, S, W)]
    pad_zeros = torch.zeros(B, D, W - 1, dtype=x.dtype, device=x.device)
    padded = torch.cat([pad_zeros, x], dim=2)
    return kernel(padded, weight, bias)
