import torch
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    """
    Reference implementation of block-scale fp8 gemm 
    Args:
        data: Input tensor to compute block-scale fp8 gemm
    Returns:
        Tensor containing output in bf16
    """
    a, b = data
    return (a @ b).to(torch.bfloat16)
