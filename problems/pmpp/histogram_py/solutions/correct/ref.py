import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    data, output = data
    output = torch.bincount(data, minlength=256)
    return output
