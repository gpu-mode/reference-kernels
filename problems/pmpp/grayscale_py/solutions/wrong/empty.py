# the nop kernel
from task import input_t, output_t
import torch


def custom_kernel(data: input_t) -> output_t:
    return torch.empty(size=(data.shape[0], data.shape[1]), device=data.device, dtype=data.dtype)
