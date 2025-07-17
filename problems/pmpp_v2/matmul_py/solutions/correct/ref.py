import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    a, b = data
    return a @ b

