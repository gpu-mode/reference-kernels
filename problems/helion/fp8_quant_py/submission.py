from task import input_t, output_t

import torch
import helion
import helion.language as hl
from pathlib import Path


# Per-shape configs: map (num_tokens, hidden_dim, group_size) to optimized helion.Config objects.
# Autotune locally for each shape, then paste the best config here.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 256, 64): helion.Config(block_sizes=[1], num_warps=1, num_stages=1),  # TODO: replace with default config or any config that passes correctness check
    (4, 512, 128): helion.Config(block_sizes=[1], num_warps=1, num_stages=1),  # TODO: replace with default config or any config that passes correctness check
    (16, 1024, 64): helion.Config(block_sizes=[1], num_warps=1, num_stages=1),  # TODO: replace with default config or any config that passes correctness check
    (1, 4096, 128): helion.Config(block_sizes=[1], num_warps=1, num_stages=1),  # TODO: replace with default config or any config that passes correctness check
    (8, 4096, 128): helion.Config(block_sizes=[1], num_warps=1, num_stages=1),  # TODO: replace with default config or any config that passes correctness check
    # Benchmark shapes
    # (1, 4096, 128) already covered above
    (16, 4096, 128): helion.Config(block_sizes=[1], num_warps=1, num_stages=1),  # TODO: replace with your autotuned config
    (256, 4096, 128): helion.Config(block_sizes=[1], num_warps=1, num_stages=1),  # TODO: replace with your autotuned config
    (256, 8192, 128): helion.Config(block_sizes=[1], num_warps=1, num_stages=1),  # TODO: replace with your autotuned config
    (4096, 7168, 128): helion.Config(block_sizes=[1], num_warps=1, num_stages=1),  # TODO: replace with your autotuned config
}


# Optional: add advanced_controls_file to your Config for extra performance (see docs).
# Autotune with autotune_search_acf to find the best ACF, then hardcode it:
#     helion.Config(..., advanced_controls_file="/opt/booster_pack/fp8_group_quant_0.acf")


# NOTE: This is an intentionally inefficient baseline implementation.
def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(
        data: torch.Tensor,       # [N, G] input rows
        scales_out: torch.Tensor,  # [N] output normalization factors
    ) -> torch.Tensor:
        nrows = data.size(0)
        ncols = hl.specialize(data.size(1))
        MAX_VAL = 448.0

        qout = torch.empty(nrows, ncols, dtype=torch.float32, device=data.device)

        for rr in hl.tile(nrows):
            row = data[rr, :].to(torch.float32)

            abs1 = torch.abs(row)
            amax1 = torch.amax(abs1, -1)
            abs2 = torch.abs(row)
            amax2 = torch.amax(abs2, -1)
            abs3 = torch.abs(row)
            amax3 = torch.amax(abs3, -1)
            amax = (amax1 + amax2 + amax3) / 3.0
            amax = torch.clamp(amax, min=1e-10)
            scale = amax / MAX_VAL

            q1 = row / scale[:, None]
            q2 = row / scale[:, None]
            q3 = row / scale[:, None]
            qout[rr, :] = (q1 + q2 + q3) / 3.0
            scales_out[rr] = scale

        return qout

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    x, x_q, x_s = data
    T, H = x.shape
    G = x_s.shape[1]
    gsz = H // G
    N = T * G

    kernel = _KERNELS[(T, H, gsz)]

    flat_in = x.reshape(N, gsz)
    flat_s = x_s.reshape(N)

    flat_q = kernel(flat_in, flat_s)

    x_q[...] = flat_q.reshape(T, H)
    x_s[...] = flat_s.reshape(T, G)
    return x_q, x_s
