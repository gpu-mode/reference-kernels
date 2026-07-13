import torch

from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    return torch.linalg.cholesky_ex(data, check_errors=False).L
