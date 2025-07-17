from utils import make_match_reference
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
    A, B, output = data
    output = A + B
    return output


def generate_input(size: int, seed: int) -> input_t:
    """
    Generates random input tensors of specified shapes.
    Returns:
        Tuple of tensors [A, B] to be added.
    """
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    A = torch.randn(
        size, size, device="cuda", dtype=torch.float16, generator=gen
    ).contiguous()
    B = torch.randn(
        size, size, device="cuda", dtype=torch.float16, generator=gen
    ).contiguous()
    C = torch.empty(size, size, device="cuda", dtype=torch.float16).contiguous()
    return (A, B, C)


check_implementation = make_match_reference(ref_kernel)
