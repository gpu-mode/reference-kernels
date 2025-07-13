from utils import make_match_reference
import torch
from task import input_t, output_t


def ref_kernel(data: input_t) -> output_t:
    """
    Reference implementation of RGB to grayscale conversion using PyTorch.
    Uses the standard coefficients: Y = 0.2989 R + 0.5870 G + 0.1140 B

    Args:
        data: RGB tensor of shape (H, W, 3) with values in [0, 1]
    Returns:
        Grayscale tensor of shape (H, W) with values in [0, 1]
    """
    data, output = data
    # Standard RGB to Grayscale coefficients
    weights = torch.tensor(
        [0.2989, 0.5870, 0.1140], device=data.device, dtype=data.dtype
    )
    output = torch.sum(data * weights, dim=-1)
    return output


def generate_input(size: int, seed: int) -> input_t:
    """
    Generates random RGB image tensor of specified size.
    Returns:
        Tensor of shape (size, size, 3) with values in [0, 1]
    """
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    x = torch.rand(
        size, size, 3, device="cuda", dtype=torch.float32, generator=gen
    ).contiguous()

    y = torch.empty(size, size, device="cuda", dtype=torch.float32).contiguous()

    return (x, y)


check_implementation = make_match_reference(ref_kernel, rtol=1e-4, atol=1e-4)
