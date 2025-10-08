import torch
from task import input_t, output_t
from utils import make_match_reference

def ref_kernel(
    data: input_t,
)->output_t:
    """
    Highly inefficient torch reference implementation of a FFAM simulated NVFP4 block-scaled GEMV.
    
    a: [l, m, k] matrix
    b: [l, 1, k] vector  
    scale_a: [l, m, k//16] blockwise scales for a
    scale_b: [l, 1, k//16] blockwise scales for b
    c: [l, m, 1] output
    
    Block size is 16 along the k dimension.
    """
    a, b, scale_a, scale_b, c = data
    
    # Make contiguous for efficiency
    a = a.contiguous()
    b = b.contiguous()

    # Get dimensions
    l, m, k = a.shape
    block_size = 16
    scale_k = k // block_size

    #reshape makes memory contiguous
    scale_a = (
        scale_a.permute(2, 0, 1)
        .unsqueeze(-1)
        .expand(l, m, scale_k, block_size)
        .reshape(l, m, scale_k * block_size)
        .permute(1, 2, 0)
    )
    scale_a = scale_a[:, :k, :]
    scale_b = (
        scale_b.permute(2, 0, 1)
        .unsqueeze(-1)
        .expand(l, 1, scale_k, block_size)
        .reshape(l, 1, scale_k * block_size)
        .permute(1, 2, 0)
    )
    scale_b = scale_b[:, :k, :]
    # scale_a = scale_a.contiguous()
    # scale_b = scale_b.contiguous()
    

    # # Apply blockwise scaling to input 'a'
    # # scale_a shape: [l, m, scale_k] -> expand to [l, m, k]
    # a_scale_expanded = scale_a.unsqueeze(-1).repeat(1, 1, 1, block_size)  # Shape: [l, m, scale_k, block_size]
    # a_scale_expanded = a_scale_expanded.reshape(l, m, scale_k * block_size)
    # a_scale_expanded = a_scale_expanded[:, :, :k]  # Handle case where k is not exactly divisible
    
    # Dequantize 'a' by applying scales, convert to float32 for computation
    a_scaled = a.to(torch.float32) * scale_a
    b_scaled = b.to(torch.float32) * scale_b
    
    # # Apply blockwise scaling to input 'b'
    # # scale_b shape: [l, 1, scale_k] -> expand to [l, 1, k]
    # b_scale_expanded = scale_b.unsqueeze(-1).repeat(1, 1, 1, block_size)  # Shape: [l, 1, scale_k, block_size]
    # b_scale_expanded = b_scale_expanded.reshape(l, 1, scale_k * block_size)
    # b_scale_expanded = b_scale_expanded[:, :, :k]  # Handle case where k is not exactly divisible
    
    # # Dequantize 'b' by applying scales, convert to float32 for computation
    # b_scaled = b.to(torch.float32) * b_scale_expanded.to(torch.float32)
    
    # Compute GEMV using batched matmul: a_scaled [l, m, k] @ b_scaled [l, 1, k] -> [l, m, 1]
    # For each batch l: a[i, :, :] @ b[i, 0, :].T
    result = torch.zeros((l, m, 1), dtype=torch.float32, device=a.device)
    for i in range(l):
        result[i, :, 0] = (a_scaled[i, :, :] @ b_scaled[i, 0, :]).to(c.dtype)
    c[...] = result.to(c.dtype)
    
    return c

def generate_input(
    m: int,
    k: int,
    l: int,
    seed: int,
):
    torch.manual_seed(seed)
    block_size = 16
    scale_k = k // block_size

    # Create fp4 input a, b tensors with LxMxK layout
    # torch.float4e2m1fn is not a standard torch dtype; use torch.uint8 as a placeholder for fp4
    a = torch.arange(l * m * k, dtype=torch.float32, device="cuda").reshape(m, k, l).to(torch.uint8)
    b = torch.arange(l * 1 * k, dtype=torch.float32, device="cuda").reshape(1, k, l).to(torch.uint8)
    # Create fp16 output tensor with LxMx1 layout
    c = torch.arange(l * m * 1, dtype=torch.float32, device="cuda").reshape(m, 1, l).to(torch.float16)

    # Create scales factor with f32 data type
    def ceil_div(a, b):
        return (a + b - 1) // b
    
    # every 16 k elements share the same scale factor
    # Set the block size for blockwise scaling
    block_size = 16
    # Compute the number of scale factors needed along k (ceil division)
    scale_k = ceil_div(k, block_size)
    # Define the shape for scale_a: [l, m, scale_k]
    scale_a_shape = (l, m, scale_k)
    # Define the shape for scale_b: [l, 1, scale_k]
    scale_b_shape = (l, 1, scale_k)
    # Permute order to match expected layout: (m, scale_k, l)
    scale_permute_order = (1, 2, 0)
    # Generate random scale factors for a, then permute to (m, scale_k, l)
    scale_a_f32 = torch.randint(1, 3, scale_a_shape, dtype=torch.float32, device="cuda").permute(scale_permute_order)
    # Generate random scale factors for b, then permute to (1, scale_k, l)
    scale_b_f32 = torch.randint(1, 3, scale_b_shape, dtype=torch.float32, device="cuda").permute(scale_permute_order)
    
    return (a, b, scale_a_f32, scale_b_f32, c)

check_implementation = make_match_reference(ref_kernel, rtol=1e-01, atol=1e-02)