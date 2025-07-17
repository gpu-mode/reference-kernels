from utils import match_reference
import torch
from task import input_t, output_t


def ref_kernel(data: input_t) -> output_t:
    """
    Reference implementation of inclusive prefix sum using PyTorch.
    Args:
        data: Input tensor to compute prefix sum on
    Returns:
        Tensor containing the inclusive prefix sum
    """
    data, output = data
    output = torch.cumsum(data.to(torch.float64), dim=0).to(torch.float64)
    return output


def generate_input(size: int, seed: int) -> input_t:
    """
    Generates random input tensor.
    Returns:
        Tensor to compute prefix sum on
    """
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    x = torch.randn(
        size, device="cuda", dtype=torch.float32, generator=gen
    ).contiguous()
    y = torch.empty(size, device="cuda", dtype=torch.float32).contiguous()
    return (x, y)


# This algorithm is very sensitive to the tolerance and the error is magnified by the input size
# The tolerance is scaled by the square root of the input size
def check_implementation(data: input_t, output: output_t) -> str:
    # Then get the size for scaling the tolerance
    n = data.numel()

    scale_factor = n**0.5  # Square root of input size
    rtol = 1e-5 * scale_factor
    atol = 1e-5 * scale_factor

    return match_reference(data, output, reference=ref_kernel, rtol=rtol, atol=atol)
