from utils import verbose_allclose
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
    # Fixed range [0, 100]
    min_val, max_val = 0, 100
    
    # Number of bins is input size / 16
    num_bins = data.shape[0] // 16

    clipped = torch.clamp(data, min_val, max_val)
    
    # Scale to bin indices
    bin_width = (max_val - min_val) / num_bins
    indices = ((clipped - min_val) / bin_width).long()
    indices = torch.clamp(indices, 0, num_bins - 1)
    
    # Count values in each bin
    return torch.bincount(indices, minlength=num_bins).to(torch.float32)

def generate_input(size: int, seed: int) -> input_t:
    """
    Generates random input tensor for histogram.
    The number of bins is automatically set to size/16.
    
    Args:
        size: Size of the input tensor (must be multiple of 16)
        seed: Random seed
    Returns:
        The input tensor with values in [0, 100]
    """
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    
    # Generate integer values between 0 and 100
    data = torch.randint(0, 101, (size,), device='cuda', dtype=torch.int32, generator=gen)
    
    # Convert to float since the histogram implementation expects float input
    return data.float().contiguous()

def check_implementation(
    data: input_t,
    output: output_t,
) -> str:
    """
    Compare custom implementation's output to the reference output.
    """
    expected = ref_kernel(data)
    reasons = verbose_allclose(output, expected)
    
    if len(reasons) > 0:
        return "mismatch found! custom implementation doesn't match reference: " + reasons[0]
    
    return ''
