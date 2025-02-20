from utils import verbose_allclose
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
    return torch.cumsum(data, dim=0)

def generate_input(size: int, seed: int) -> input_t:
    """
    Generates random input tensor.
    Returns:
        Tensor to compute prefix sum on
    """
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    return torch.randn(size, device='cuda', dtype=torch.float32, generator=gen).contiguous()

def check_implementation(
    data: input_t,
    output: output_t,
) -> str:
    expected = ref_kernel(data)
    reasons = verbose_allclose(output, expected, rtol=1e-5, atol=1e-5)
    
    if len(reasons) > 0:
        return "mismatch found! custom implementation doesn't match reference: " + reasons[0]
    
    return '' 