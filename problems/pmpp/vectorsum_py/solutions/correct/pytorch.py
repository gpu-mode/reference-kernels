import torch
import triton
import triton.language as tl
from task import input_t, output_t


def _custom_kernel(data: input_t) -> output_t:
    data, output = data
    output = data.sum()
    return output


# Compile the kernel for better performance
custom_kernel = torch.compile(_custom_kernel, mode="reduce-overhead")
