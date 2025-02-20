from utils import verbose_allclose
import torch
from task import input_t, output_t, HistogramSpec

def ref_kernel(data: input_t, spec: HistogramSpec) -> output_t:
    """
    Reference implementation of histogram using PyTorch.
    Args:
        data: Input tensor to compute histogram on
        spec: Histogram specifications (num_bins, min_val, max_val)
    Returns:
        Tensor containing bin counts
    """
    # Clip values to range
    clipped = torch.clamp(data, spec.min_val, spec.max_val)
    
    # Scale to bin indices
    bin_width = (spec.max_val - spec.min_val) / spec.num_bins
    indices = ((clipped - spec.min_val) / bin_width).long()
    indices = torch.clamp(indices, 0, spec.num_bins - 1)
    
    # Count values in each bin
    return torch.bincount(indices, minlength=spec.num_bins).to(torch.float32)

def generate_input(size: int, seed: int) -> input_t:
    """
    Generates random input tensor with values roughly in [0, 1].
    Returns:
        Tensor to compute histogram on
    """
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    # Generate values with normal distribution for interesting histograms
    return torch.randn(size, device='cuda', dtype=torch.float32, generator=gen).contiguous()

def check_implementation(
    data: input_t,
    spec: HistogramSpec,
    output: output_t,
) -> str:
    expected = ref_kernel(data, spec)
    reasons = verbose_allclose(output, expected)
    
    if len(reasons) > 0:
        return "mismatch found! custom implementation doesn't match reference: " + reasons[0]
    
    return '' 