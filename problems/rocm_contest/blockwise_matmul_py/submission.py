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
    a, b, a_scale, b_scale = data
    block_shape_n, block_shape_k = block_shape
    scale_n, scale_k = b_scale.shape
    # Apply scaling to input 'a'
    a = a.to(a_scale.dtype).view(m, k//block_shape_k, block_shape_k) * a_scale.unsqueeze(-1)
    a = a.view(m, k)    
    # Apply scaling to input 'b'
    b_scale = (
        b_scale.view(-1, 1)
        .repeat(1, block_shape_n * block_shape_k)
        .view(scale_n, scale_k, block_shape_n, block_shape_k)
        .permute(0, 2, 1, 3)  # Reorder dimensions: [scale_n, blk_n, scale_k, blk_k]
        .reshape(scale_n * block_shape_n, scale_k * block_shape_k)
    )
    b_scale = b_scale[:n, :k]
    b = b.to(b_scale.dtype) * b_scale    
    return (a @ b).to(torch.bfloat16)
