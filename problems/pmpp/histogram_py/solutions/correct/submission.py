import torch
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
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
    
    # Clip values to range
    clipped = torch.clamp(data, min_val, max_val)
    
    # Scale to bin indices
    bin_width = (max_val - min_val) / num_bins
    indices = ((clipped - min_val) / bin_width).long()
    indices = torch.clamp(indices, 0, num_bins - 1)
    
    # Count values in each bin
    return torch.bincount(indices, minlength=num_bins).to(torch.float32)
