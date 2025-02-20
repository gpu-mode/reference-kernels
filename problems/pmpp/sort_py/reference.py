from utils import verbose_allclose
import torch
from task import input_t, output_t

def ref_kernel(data: input_t) -> output_t:
    """
    Reference implementation of sort using PyTorch.
    Args:
        data: Input tensor to be sorted
    Returns:
        Sorted tensor
    """
    return torch.sort(data)[0]

def generate_input(size: int, seed: int) -> input_t:
    """
    Generates random input tensor.
    Returns:
        Tensor to be sorted
    """
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    return torch.randn(size, device='cuda', dtype=torch.float32, generator=gen).contiguous()

def check_implementation(
    data: input_t,
    output: output_t,
) -> str:
    expected = ref_kernel(data)
    reasons = verbose_allclose(output, expected)
    
    if len(reasons) > 0:
        return "mismatch found! custom implementation doesn't match reference: " + reasons[0]
    
    return '' 