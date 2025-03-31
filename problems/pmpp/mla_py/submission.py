import torch
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    """
    Reference implementation of mla fused attention using PyTorch.
    Args:
        data: Input tensor q, k_cache, v_cache to mla fused attention decode
    Returns:
        Tensor include attention output
    """
    return torch.cumsum(data, dim=0)
