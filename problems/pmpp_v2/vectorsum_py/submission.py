#!POPCORN leaderboard vectorsum_py

import torch
import triton
import triton.language as tl
from task import input_t, output_t


@triton.jit
def sum_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Parallel reduction kernel that sums elements in chunks.
    Each thread block reduces BLOCK_SIZE elements.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute local reduction
    block_sum = tl.sum(x, axis=0)

    # Store the partial sum
    tl.atomic_add(output_ptr, block_sum)


def _custom_kernel(data: input_t) -> output_t:
    """
    Performs parallel reduction to compute sum of all elements.
    Args:
        data: Input tensor to be reduced
    Returns:
        Tensor containing the sum of all elements
    """
    n_elements = data.numel()
    output = torch.zeros(1, device=data.device, dtype=data.dtype)

    # Configure kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch kernel
    sum_kernel[grid](
        data,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output[0]


# Compile the kernel for better performance
custom_kernel = torch.compile(_custom_kernel, mode="reduce-overhead")
