import torch
from task import input_t, output_t


def _custom_kernel(data: input_t) -> output_t:
    return torch.sort(data)[0]


custom_kernel = torch.compile(_custom_kernel, mode="reduce-overhead")
