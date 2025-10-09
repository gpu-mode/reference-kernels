import torch
from task import input_t, output_t
from utils import make_match_reference

def ceil_div(a, b):
    """Helper function for ceiling division"""
    return (a + b - 1) // b

def ref_kernel(
    data: input_t,
) -> output_t:
    """
    PyTorch reference implementation of NVFP4 block-scaled GEMV.
    
    This simulates the GEMV operation: C = A @ b
    where A and b are block-scaled with FP4 values and FP8 scale factors.
    
    Tensor shapes (MxKxL layout):
        a: [m, k, l] - Input matrix in FP4
        b: [1, k, l] - Input vector in FP4  
        scale_a: [m, k, l] - Expanded blockwise scales for a in FP32
        scale_b: [1, k, l] - Expanded blockwise scales for b in FP32
        c: [m, 1, l] - Output vector in FP16
    
    where:
        m: number of rows in A
        k: number of columns in A (must be multiple of block_size)
        l: batch size
    
    The reference implementation follows the pattern:
        res_a = einsum("mkl,mkl->mkl", a_ref, sfa_ref)
        res_b = einsum("nkl,nkl->nkl", b_ref, sfb_ref)  # n=1 for GEMV
        ref = einsum("mkl,nkl->mnl", res_a, res_b)
    """
    a, b, scale_a, scale_b, c = data
    
    # Get dimensions from MxKxL layout
    m, k, l = a.shape
    n = 1  # GEMV: N dimension is always 1

    # Convert to f32 for reference computation
    a_ref = a.to(torch.float32)
    b_ref = b.to(torch.float32)
    sfa_ref = scale_a.to(torch.float32)
    sfb_ref = scale_b.to(torch.float32)
    
    # Apply blockwise scaling: elementwise multiplication
    # This simulates NVFP4 GEMV via 2 FFMA based elementwise multiplication 
    # and 1 FFMA based matmul computations
    res_a = a_ref * sfa_ref  # [m, k, l]
    res_b = b_ref * sfb_ref  # [1, k, l]
    
    # Compute batched GEMV: C[m, n, l] = A[m, k, l] @ B[n, k, l]
    # For each batch: c[:, :, i] = a[:, :, i] @ b[0, :, i].T
    result = torch.zeros((m, n, l), dtype=torch.float32, device=a.device)
    for i in range(l):
        # res_a[:, :, i] is [m, k], res_b[0, :, i] is [k]
        # matmul gives [m], reshape to [m, 1]
        result[:, 0, i] = res_a[:, :, i] @ res_b[0, :, i]
    
    # Store result in output tensor
    c[...] = result.to(c.dtype)
    return c


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
    block_size = 16
    n = 1  # GEMV: N dimension is always 1
    scale_k = ceil_div(k, block_size)

    # Create input tensors A, b following the MxKxL memory layout
    # This matches: cutlass_torch.matrix(l, m, k, False, cutlass.Float32)
    # which creates tensors with contiguous k dimension (stride-1)
    
    # Generate random FP32 values, then convert to uint8 (FP4 placeholder)
    # Shape transformations: (l, m, k) -> permute to (m, k, l) for MxKxL layout
    a = torch.randn(l, m, k, dtype=torch.float32, device="cuda")
    a = a.permute(1, 2, 0).contiguous().to(torch.uint8)  # [m, k, l]
    
    b = torch.randn(l, n, k, dtype=torch.float32, device="cuda")
    b = b.permute(1, 2, 0).contiguous().to(torch.uint8)  # [1, k, l]
    
    # Create output tensor C in FP16 with MxNxL layout (N=1 for GEMV)
    c = torch.zeros(l, m, n, dtype=torch.float32, device="cuda")
    c = c.permute(1, 2, 0).contiguous().to(torch.float16)  # [m, 1, l]
    
    # Create scale factors with FP32 data type
    # Original ref_shape is (l, mn, sf_k), then permuted to (mn, sf_k, l)
    ref_shape = (l, m, scale_k)
    ref_permute_order = (1, 2, 0)  # Permute from LxMxScaleK to MxScaleKxL
    
    # Generate random scale factors in range [1, 3) for better numerical stability
    scale_a_sf = torch.randint(1, 3, ref_shape, dtype=torch.float32, device="cuda")  # [1, 3)
    scale_a_sf = scale_a_sf.permute(ref_permute_order).contiguous()  # [m, scale_k, l]
    
    ref_shape_b = (l, n, scale_k)
    scale_b_sf = torch.randint(1, 3, ref_shape_b, dtype=torch.float32, device="cuda")  # [1, 3)
    scale_b_sf = scale_b_sf.permute(ref_permute_order).contiguous()  # [n, scale_k, l]
    
    # Expand scale factors from [m, scale_k, l] to [m, k, l]
    # This matches the expansion done in nvfp4_gemv_cute_layout.py lines 320-328
    # The pattern: permute -> unsqueeze -> expand -> reshape -> permute -> prune
    scale_a_expanded = (
        scale_a_sf.permute(2, 0, 1)  # [l, m, scale_k]
        .unsqueeze(-1)  # [l, m, scale_k, 1]
        .expand(l, m, scale_k, block_size)  # [l, m, scale_k, block_size]
        .reshape(l, m, scale_k * block_size)  # [l, m, k]
        .permute(*ref_permute_order)  # [m, k, l]
    )
    scale_a_expanded = scale_a_expanded[:, :k, :]  # Prune to exact k
    
    scale_b_expanded = (
        scale_b_sf.permute(2, 0, 1)  # [l, n, scale_k]
        .unsqueeze(-1)  # [l, n, scale_k, 1]
        .expand(l, n, scale_k, block_size)  # [l, n, scale_k, block_size]
        .reshape(l, n, scale_k * block_size)  # [l, n, k]
        .permute(*ref_permute_order)  # [n, k, l]
    )
    scale_b_expanded = scale_b_expanded[:, :k, :]  # Prune to exact k
    
    return (a, b, scale_a_expanded, scale_b_expanded, c)


check_implementation = make_match_reference(ref_kernel, rtol=1e-02, atol=1e-01)
