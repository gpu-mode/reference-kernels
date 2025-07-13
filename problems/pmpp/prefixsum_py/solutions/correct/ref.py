import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    output, data = data
    output = torch.cumsum(data, dim=0)
    return output
