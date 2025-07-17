import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """
    Reference implementation of histogram using PyTorch.
    Args:
        data: tensor of shape (size,)
    Returns:
        Tensor containing bin counts
    """
    data, output = data
    # Compute histogram with 256 bins
    output = torch.bincount(data, minlength=256)
    return output
