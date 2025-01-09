from typing import List

import torch


def custom_kernel(xs: List[torch.Tensor], dim: int = -1) -> List[torch.Tensor]:
    """
    Custom implementation of the Softmax function.
    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension along which to apply softmax.
    Returns:
        torch.Tensor: Tensor after applying Softmax.
    """
    res = []
    for x in xs:
        # Shift for numerical stability
        x_shifted = x - torch.max(x, dim=dim, keepdim=True).values
        exp_x = torch.exp(x_shifted)
        softmax = exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
        res.append(softmax)

    return res
