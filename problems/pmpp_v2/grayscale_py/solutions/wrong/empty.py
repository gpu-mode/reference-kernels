# the nop kernel
from task import input_t, output_t
import torch


def custom_kernel(data: input_t) -> output_t:
    _, output = data
    return output
