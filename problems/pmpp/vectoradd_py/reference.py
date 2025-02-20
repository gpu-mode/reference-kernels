from utils import verbose_allclose
import torch
from task import input_t, output_t

def ref_kernel(data: input_t) -> output_t:
    """
    Reference implementation of vector addition using PyTorch.
    Args:
        data: Tuple of tensors [A, B] to be added.
    Returns:
        Tensor containing element-wise sums.
    """
    A, B = data
    return A + B

def generate_input(size: int, seed: int) -> input_t:
    """
    Generates random input tensors of specified shapes.
    Returns:
        Tuple of tensors [A, B] to be added.
    """
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    A = torch.randn(size, size, device='cuda', dtype=torch.float16, generator=gen).contiguous()
    B = torch.randn(size, size, device='cuda', dtype=torch.float16, generator=gen).contiguous()
    return (A, B)

def check_implementation(
    data: input_t,
    output: output_t,
) -> bool:
    expected = ref_kernel(data)
    reasons = verbose_allclose(output, expected)
    
    if len(reasons) > 0:
        return "mismatch found! custom implementation doesn't match reference: " + reasons[0]
    
    return ''
