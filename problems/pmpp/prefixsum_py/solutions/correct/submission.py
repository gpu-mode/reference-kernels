import torch
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    """
    Reference implementation of inclusive prefix sum using PyTorch.
    Args:
        data: Input tensor to compute prefix sum on
    Returns:
        Tensor containing the inclusive prefix sum
    """
    return torch.cumsum(data, dim=0)