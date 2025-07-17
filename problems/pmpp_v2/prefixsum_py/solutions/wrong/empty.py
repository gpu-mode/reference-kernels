# the nop kernel
from task import input_t, output_t
import torch


def custom_kernel(data: input_t) -> output_t:
    return torch.empty(size=data.shape, device=data.device, dtype=data.dtype)
