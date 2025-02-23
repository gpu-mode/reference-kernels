from task import input_t, output_t
import torch
import torch.nn.functional as F


def custom_kernel(data: input_t) -> output_t:
    """
    Implementation of 2D convolution using PyTorch with no padding and no striding.
    Args:
        data: Tuple of (input tensor, kernel tensor)
        spec: Convolution specifications
    Returns:
        Output tensor after convolution
    """
    input_tensor, kernel = data
    return F.conv2d(
        input_tensor, 
        kernel,
        stride=1,
        padding=0
    )