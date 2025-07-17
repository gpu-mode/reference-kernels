from utils import make_match_reference
import torch
import torch.nn.functional as F
from task import input_t, output_t


def ref_kernel(data: input_t) -> output_t:
    """
    Reference implementation of 2D convolution using PyTorch.
    Args:
        data: Tuple of (input tensor, kernel tensor)
    Returns:
        Output tensor after convolution
    """
    input_tensor, kernel = data
    return F.conv2d(
        input_tensor, 
        kernel,

        # No padding and no striding
        # TODO: Can revisit this in future problems
        stride=1,
        padding=0
    )


def generate_input(size: int, kernelsize: int, channels: int, batch: int, seed: int) -> input_t:
    """
    Generates random input and kernel tensors.
    Returns:
        Tuple of (input tensor, kernel tensor)
    """
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    
    # Generate input tensor: [batch, in_channels, height, width]
    input_tensor = torch.randn(
        batch, channels, size, size,
        device='cuda', 
        dtype=torch.float32, 
        generator=gen
    ).contiguous()
    
    # Generate kernel tensor: [out_channels, in_channels, kernel_height, kernel_width]
    # Here we use same number of output channels as input channels for simplicity
    kernel = torch.randn(
        channels, channels, kernelsize, kernelsize,
        device='cuda',
        dtype=torch.float32,
        generator=gen
    ).contiguous()
    
    return (input_tensor, kernel)


check_implementation = make_match_reference(ref_kernel, rtol=1e-3, atol=1e-3)
