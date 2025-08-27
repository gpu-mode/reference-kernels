from utils import verbose_allequal, DeterministicContext
import torch
from task import input_t, output_t


def ref_kernel(data: input_t) -> output_t:
    """
    Reference implementation of histogram using PyTorch.
    Args:
        data: tensor of shape (size,)
    Returns:
        Tensor containing bin counts
    """
    with DeterministicContext():
        data, output = data
        # Count values in each bin
        output[...] = torch.bincount(data, minlength=256)
        return output


def generate_input(size: int, contention: float, seed: int) -> input_t:
    """
    Generates random input tensor for histogram.

    Args:
        size: Size of the input tensor (must be multiple of 16)
        contention: float in [0, 100], specifying the percentage of identical values
        seed: Random seed
    Returns:
        The input tensor with values in [0, 255]
    """
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    
    # Generate integer values between 0 and 256
    data = torch.randint(0, 256, (size,), device='cuda', dtype=torch.uint8, generator=gen)

    # make one value appear quite often, increasing the chance for atomic contention
    evil_value = torch.randint(0, 256, (), device='cuda', dtype=torch.uint8, generator=gen)
    evil_loc = torch.rand((size,), device='cuda', dtype=torch.float32, generator=gen) < (contention / 100.0)
    data[evil_loc] = evil_value

    output = torch.empty(256, device='cuda', dtype=torch.int64).contiguous()

    return data.contiguous(), output


def check_implementation(data, output):
    expected = ref_kernel(data)
    reasons = verbose_allequal(output, expected)

    if len(reasons) > 0:
        return False, "mismatch found! custom implementation doesn't match reference: " + " ".join(reasons)

    return True, ''

