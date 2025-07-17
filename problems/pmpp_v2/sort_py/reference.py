from utils import make_match_reference
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


def generate_input(size: int, seed: int) -> torch.Tensor:
    """
    Generates random input tensor where elements are drawn from different distributions.
    
    Args:
        size: Total size of the final 1D tensor
        seed: Base seed for random generation
    
    Returns:
        1D tensor of size `size` containing flattened values from different distributions
    """
    # Calculate dimensions for a roughly square 2D matrix
    rows = int(size ** 0.5)  # Square root for roughly square shape
    cols = (size + rows - 1) // rows  # Ceiling division to ensure total size >= requested size
    
    gen = torch.Generator(device='cuda')
    result = torch.empty((rows, cols), device='cuda', dtype=torch.float32)
    
    # Different seed for each row!
    for i in range(rows):
        row_seed = seed + i
        gen.manual_seed(row_seed)
        
        # Generate values for this row with mean=row_seed
        result[i, :] = torch.randn(cols, device='cuda', dtype=torch.float32, generator=gen) + row_seed
    
    # Flatten and trim to exact size requested
    return result.flatten()[:size].contiguous()


check_implementation = make_match_reference(ref_kernel)
