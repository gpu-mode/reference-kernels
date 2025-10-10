import torch
from task import input_t, output_t
from utils import make_match_reference

block_size = 16

def ceil_div(a, b):
    """Helper function for ceiling division"""
    return (a + b - 1) // b

def ref_kernel(
    data: input_t,
) -> output_t:
    """
    PyTorch reference implementation of NVFP4 block-scaled GEMV.
    This is a very slow reference implementation to show the computation details
    of a block-scaled GEMV.
    
    This simulates the GEMV operation: C = A @ b
    where A and b are block-scaled with FP4 values and FP8 scale factors.
    """
    a, b, scale_a, scale_b, c = data
    
    # Get dimensions from MxKxL layout
    m, k, l = a.shape
    n = 1  # GEMV: N dimension is always 1

    scale_k = ceil_div(k, block_size)

    # Extend scale factor tensor from [m, scale_k, l] to [m, k, l]
    ref_permute_order = (1, 2, 0)
    scale_a = (
        scale_a.permute(2, 0, 1)
        .unsqueeze(-1)
        .expand(l, m, scale_k, block_size)
        .reshape(l, m, scale_k * block_size)
        .permute(*ref_permute_order)
    )
    # prune to mkl for reference check.
    scale_a = scale_a[:, :k, :]

    scale_b = (
        scale_b.permute(2, 0, 1)
        .unsqueeze(-1)
        .expand(l, n, scale_k, block_size)
        .reshape(l, n, scale_k * block_size)
        .permute(*ref_permute_order)
    )
    # prune to mkl for reference check.
    scale_b = scale_b[:, :k, :]

    # Convert to f32 for computation
    # Apply blockwise scaling: elementwise multiplication
    # This simulates NVFP4 GEMV via 2 FFMA based elementwise multiplication 
    # and 1 FFMA based matmul computations
    res_a = a.to(torch.float32) * scale_a.cuda()  # [m, k, l]
    res_b = b.to(torch.float32) * scale_b.cuda()  # [1, k, l]
    
    # Compute batched GEMV: C[m, n, l] = A[m, k, l] @ B[n, k, l]
    for i in range(c.shape[2]):
        # matmul gives [m], convert to c.dtype then assign to [m, 1]
        acc = res_a[:, :, i] @ res_b[0, :, i]
        c[:, 0, i] = acc.to(torch.float16)
    return c


# Helper function to create reference scale factor tensor SFA/SFB
# for 1x16 block scaled wise use case and follow the layout requirement
# defined in https://docs.nvidia.com/cuda/cublas/index.html?highlight=fp4#d-block-scaling-factors-layout
def create_scale_factor_tensor(l, mn, k, block_size):
    scale_k = ceil_div(k, block_size)
    ref_shape = (l, mn, scale_k)
    ref_permute_order = (1, 2, 0)

    # Create f32 ref torch tensor (cpu)
    ref_f32_torch_tensor_cpu = torch.randint(
        1, 3, ref_shape, dtype=torch.float32
    ).permute(*ref_permute_order)
    return ref_f32_torch_tensor_cpu

def generate_input(
    m: int,
    k: int,
    l: int,
    seed: int,
):
    """
    Generate input tensors for NVFP4 block-scaled GEMV.
    
    This follows the pattern from nvfp4_gemv_cute_layout.py for tensor preparation.
    
    Args:
        m: Number of rows in matrix A
        k: Number of columns in A (and length of vector b)
        l: Batch size
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (a, b, scale_a, scale_b, c) where:
            a: [m, k, l] - Input matrix in FP4 (simulated with uint8)
            b: [1, k, l] - Input vector in FP4 (simulated with uint8)
            scale_a: [m, k, l] - Expanded scale factors for a in FP32
            scale_b: [1, k, l] - Expanded scale factors for b in FP32
            c: [m, 1, l] - Output vector in FP16
    """
    torch.manual_seed(seed)
    n = 1  # GEMV: N dimension is always 1
    
    # Generate random FP32 values, then convert to uint8 (FP4 placeholder)
    a = torch.randint(0, 2, (l, m, k), dtype=torch.uint8, device="cuda").permute(1, 2, 0)
    b = torch.randint(1, 3, (l, n, k), dtype=torch.uint8, device="cuda").permute(1, 2, 0)
    c = torch.randn((l, n, m), dtype=torch.float16, device="cuda").permute(2, 1, 0)
    
    # Create scale factors with FP32 data type
    scale_a = create_scale_factor_tensor(l, m, k, block_size)
    scale_b = create_scale_factor_tensor(l, 1, k, block_size)
    
    return (a, b, scale_a, scale_b, c)


check_implementation = make_match_reference(ref_kernel, rtol=1e-01, atol=1e-02)
