import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    a, b, c = data
    c = (a.to(torch.bfloat16) @ b.to(torch.bfloat16)).to(a.dtype)
    return c
