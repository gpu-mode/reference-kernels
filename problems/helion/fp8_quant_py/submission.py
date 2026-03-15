from task import input_t, output_t

import torch
import helion
import helion.language as hl
from pathlib import Path


# Per-shape configs: map (num_tokens, hidden_dim, group_size) to optimized helion.Config objects.
# Autotune locally for each shape, then paste the best config here.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 256, 64):     helion.Config(block_sizes=[1],  num_warps=1, num_stages=1),
    (4, 512, 128):    helion.Config(block_sizes=[2],  num_warps=2, num_stages=1),
    (16, 1024, 64):   helion.Config(block_sizes=[4],  num_warps=2, num_stages=1),
    (1, 4096, 128):   helion.Config(block_sizes=[1],  num_warps=1, num_stages=1),
    (8, 4096, 128):   helion.Config(block_sizes=[4],  num_warps=2, num_stages=1),
    # Benchmark shapes
    (16, 4096, 128):  helion.Config(block_sizes=[4],  num_warps=4, num_stages=2),
    (256, 4096, 128): helion.Config(block_sizes=[8],  num_warps=8, num_stages=2),
    (256, 8192, 128): helion.Config(block_sizes=[8],  num_warps=8, num_stages=2),
    (4096, 7168, 128):helion.Config(block_sizes=[16], num_warps=8, num_stages=2),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(
        data: torch.Tensor,        # [N, G] float32
        scales_out: torch.Tensor,  # [N]
    ) -> torch.Tensor:
        nrows = data.size(0)
        ncols = hl.specialize(data.size(1))
        MAX_VAL = 448.0
        qout = torch.empty(nrows, ncols, dtype=torch.float32, device=data.device)
        for rr in hl.tile(nrows):
            row = data[rr, :].to(torch.float32)
            amax = torch.amax(torch.abs(row), dim=-1).clamp(min=1e-10)
            scale = amax / MAX_VAL
            qout[rr, :] = torch.clamp(row / scale[:, None], -MAX_VAL, MAX_VAL)
            scales_out[rr] = scale
        return qout

    return kernel


# Create a kernel for each shape based on the configs above.
_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


# TODO Fuse reshapes into kernel. Reshape is a no-op for the GPU, but requires
# copy kernel in pytorch eager mode.

def custom_kernel(data: input_t) -> output_t:
    x, x_q, x_s = data
    # num_tokens, hidden_dim = x.shape
    T, H = x.shape
    # num_groups = x_s.shape[1]
    G = x_s.shape[1]
    # group_size = hidden_dim // num_groups
    gsz = H // G
    # merge num_tokens and num_groups into a single dimension for the kernel
    N = T * G

    # Select the appropriate kernel based on the input shape.
    kernel = _KERNELS[(T, H, gsz)]

    # Reshape inputs and scales for the kernel
    # inputs is input argument
    # scales is output argument, but we need to pass it in as an argument
    flat_in = x.reshape(N, gsz)
    flat_s = x_s.reshape(N)

    # return value is quantized output
    flat_q = kernel(flat_in, flat_s)

    # Reshape outputs back to original shapes
    x_q[...] = flat_q.reshape(T, H)
    x_s[...] = flat_s.reshape(T, G)
    return x_q, x_s
