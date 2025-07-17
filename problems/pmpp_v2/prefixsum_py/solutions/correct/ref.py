import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    data, output = data
    output[...] = torch.cumsum(data, dim=0)
    return output
