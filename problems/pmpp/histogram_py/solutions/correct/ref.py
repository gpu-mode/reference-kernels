import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    return torch.bincount(data, minlength=256)
