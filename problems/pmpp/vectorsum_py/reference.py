from utils import make_match_reference
import torch
from task import input_t, output_t


def ref_kernel(data: input_t) -> output_t:
    """
    Reference implementation of vector sum reduction using PyTorch.
    Args:
        data: Input tensor to be reduced
    Returns:
        Tensor containing the sum of all elements
    """
    data, output = data
    # Let's be on the safe side here, and do the reduction in 64 bit
    output = data.to(torch.float64).sum().to(torch.float32)
    return output


def generate_input(size: int, seed: int) -> input_t:
    """
    Generates random input tensor of specified shape with random offset and scale.
    The data is first generated as standard normal, then scaled and offset
    to prevent trivial solutions.

    Returns:
        Tensor to be reduced
    """
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    # Generate base random data
    data = torch.randn(
        size, device="cuda", dtype=torch.float32, generator=gen
    ).contiguous()

    # Generate random offset and scale (using different seeds to avoid correlation)
    offset_gen = torch.Generator(device="cuda")
    offset_gen.manual_seed(seed + 1)
    scale_gen = torch.Generator(device="cuda")
    scale_gen.manual_seed(seed + 2)

    # Generate random offset between -100 and 100
    offset = (torch.rand(1, device="cuda", generator=offset_gen) * 200 - 100).item()
    # Generate random scale between 0.1 and 10
    scale = (torch.rand(1, device="cuda", generator=scale_gen) * 9.9 + 0.1).item()

    # Apply scale and offset
    input_tensor = (data * scale + offset).contiguous()
    output_tensor = torch.empty(1, device="cuda", dtype=torch.float32)
    return (input_tensor, output_tensor)


check_implementation = make_match_reference(ref_kernel)
