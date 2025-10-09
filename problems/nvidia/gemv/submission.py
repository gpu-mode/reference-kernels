import cuda.bindings.driver as cuda

import torch
from task import input_t, output_t

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.blockscaled_layout as blockscaled_utils

# Kernel configuration parameters
mma_tiler_mnk = (128, 1, 64)  # Tile sizes for M, N, K dimensions
ab_dtype = cutlass.Float4E2M1FN  # FP4 data type for A and B
sf_dtype = cutlass.Float8E8M0FNU  # FP8 data type for scale factors
c_dtype = cutlass.Float16  # FP16 output type
block_size = 16  # Scale factor block size (16 elements share one scale)
threads_per_cta = 128  # Number of threads per CUDA thread block

def ceil_div(a, b):
    """Helper function for ceiling division"""
    return (a + b - 1) // b


@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_tensor: cute.Tensor,
    sf_mma_tensor: cute.Tensor,
):
    """
    Convert scale factor tensor from reference MxKxL layout to MMA layout.
    
    This follows the cuBLAS block-scaling factors layout specification:
    https://docs.nvidia.com/cuda/cublas/index.html?highlight=fp4#d-block-scaling-factors-layout

    """
    # sf_mma_tensor has flattened shape (32, 4, rest_m, 4, rest_k, l)
    # Group modes to ((32, 4, rest_m), (4, rest_k), l) for hierarchical indexing
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    
    # Copy data from reference layout to MMA layout
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]


@cute.kernel
def kernel(
    mA_mkl: cute.Tensor,
    mB_nkl: cute.Tensor,
    mSFA_mkl: cute.Tensor,
    mSFB_nkl: cute.Tensor,
    mC_mnl: cute.Tensor,
):
    """
    GPU device kernel for NVFP4 block-scaled GEMV computation.
    
    This kernel simulates NVFP4 computation using FFMA (Fused Multiply-Add).
    Each thread computes one element of the output vector.
    
    Args:
        mA_mkl: Input matrix A in MxKxL layout (FP4)
        mB_nkl: Input vector b in NxKxL layout where N=1 (FP4)
        mSFA_mkl: Scale factors for A (FP8)
        mSFB_nkl: Scale factors for b (FP8)
        mC_mnl: Output vector c in MxNxL layout where N=1 (FP16)
    """
    # Get thread and block indices
    bidx, bidy, bidz = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    # Tile input tensors according to MMA configuration
    # Each tile processes mma_tiler_mnk elements
    
    # Tile A: shape becomes (bM, bK, RestM, RestK, RestL)
    # bM = (32, 4) for Tensor Core, bK = (16, 4) for scale blocks
    gA_mkl = cute.local_tile(
        mA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    
    # Tile scale factors for A with same pattern
    gSFA_mkl = cute.local_tile(
        mSFA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    
    # Tile B: shape becomes (bN, bK, RestN, RestK, RestL) where N=1
    gB_nkl = cute.local_tile(
        mB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    
    # Tile scale factors for B with same pattern
    gSFB_nkl = cute.local_tile(
        mSFB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    
    # Tile output C: shape becomes (bM, bN, RestM, RestN, RestL)
    gC_mnl = cute.local_tile(
        mC_mnl, cute.slice_(mma_tiler_mnk, (None, None, 0)), (None, None, None)
    )

    # Each thread computes one output element
    # Index into output tile using thread ID and block coordinates
    tCgC = gC_mnl[tidx, None, bidx, bidy, bidz]
    tCgC = cute.make_tensor(tCgC.iterator, 1)
    res = cute.zeros_like(tCgC, cutlass.Float32)

    # Create Tensors for tile views with proper shapes
    k_tile_cnt = gA_mkl.layout[3].shape
    for k_tile in range(k_tile_cnt):
        # Extract this thread's slice of A, B, and scale factors for current K tile
        tAgA = gA_mkl[tidx, None, bidx, k_tile, bidz]
        tBgB = gB_nkl[None, None, bidy, k_tile, bidz]
        tAgSFA = gSFA_mkl[tidx, None, bidx, k_tile, bidz]
        tBgSFB = gSFB_nkl[None, None, bidy, k_tile, bidz]

        # Load A/B/SFA/SFB tile from global memory
        a_val_nvfp4 = tAgA.load()
        b_val_nvfp4 = tBgB.load()
        sfa_val_fp8 = tAgSFA.load()
        sfb_val_fp8 = tBgSFB.load()

        # Convert to FP32 for FFMA computation
        a_val = a_val_nvfp4.to(cutlass.Float32)
        b_val = b_val_nvfp4.to(cutlass.Float32)
        sfa_val = sfa_val_fp8.to(cutlass.Float32)
        sfb_val = sfb_val_fp8.to(cutlass.Float32)

        # Compute block-scaled dot product
        # Each scale factor applies to block_size consecutive elements
        for i in cutlass.range_constexpr(mma_tiler_mnk[2] // block_size):
            for j in cutlass.range_constexpr(block_size):
                # Accumulate: res += (a * scale_a) * (b * scale_b)
                res += (
                    a_val[i * block_size + j]
                    * sfa_val[i]
                    * b_val[i * block_size + j]
                    * sfb_val[i]
                )
    
    # Store result back to global memory in FP16
    tCgC.store(res.to(cutlass.Float16))
    return


@cute.jit
def my_kernel(
    a_tensor: cute.Tensor,
    b_tensor: cute.Tensor,
    sfa_tensor: cute.Tensor,
    sfb_tensor: cute.Tensor,
    c_tensor: cute.Tensor,
):
    """
    Host-side JIT function to prepare tensors and launch GPU kernel.
    
    This function:
    1. Converts scale factor tensors to the correct MMA layout
    2. Computes grid dimensions based on tensor shapes
    3. Launches the CUDA kernel
    
    Args:
        a_tensor: Input matrix A (CuTe tensor)
        b_tensor: Input vector b (CuTe tensor)
        sfa_tensor: Scale factors for A (CuTe tensor)
        sfb_tensor: Scale factors for B (CuTe tensor)
        c_tensor: Output vector c (CuTe tensor)
    """
    # Convert scale factor tensors to MMA layout
    # The layout matches Tensor Core requirements: (((32, 4), REST_M), ((SF_K, 4), REST_K), (1, REST_L))
    sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
        a_tensor.shape, block_size
    )
    sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)
    
    sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
        b_tensor.shape, block_size
    )
    sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)
    
    # Compute grid dimensions
    # Grid is (M_blocks, 1, L) where:
    # - M_blocks = ceil(M / 128) to cover all output rows
    # - N=1 for GEMV (middle dimension)
    # - L = batch size
    grid = (
        cute.ceil_div(c_tensor.shape[0], 128),
        1,
        c_tensor.shape[2],
    )
    
    # Launch the CUDA kernel
    kernel(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor).launch(
        grid=grid,
        block=[threads_per_cta, 1, 1],
        cluster=(1, 1, 1),
    )
    return


def create_scale_factor_tensor(l, mn, k, sf_vec_size, dtype, ref_sf):
    """
    Create scale factor tensor in MMA layout for CuTe kernel.
    
    This function converts a reference scale tensor from MxScaleKxL layout
    to the MMA-compatible layout required by Tensor Cores.
    
    Args:
        l: Batch size
        mn: M or N dimension (number of rows for A, 1 for b)
        k: K dimension (full, not scale_k)
        sf_vec_size: Scale factor block size (16)
        dtype: Target CuTe data type (e.g., cutlass.Float8E8M0FNU)
        ref_sf: Reference scale tensor in [mn, scale_k, l] layout (PyTorch tensor)
    
    Returns:
        Tuple of (ref_expanded, cute_tensor, cute_torch_tensor):
            - ref_expanded: Expanded reference tensor in [mn, k, l] layout for CPU validation
            - cute_tensor: CuTe tensor with MMA layout
            - cute_torch_tensor: Underlying PyTorch tensor
    """
    sf_k = ceil_div(k, sf_vec_size)

    atom_m = (32, 4)
    atom_k = 4
    mma_shape = (
        l,  # batch size
        ceil_div(mn, atom_m[0] * atom_m[1]),
        ceil_div(sf_k, atom_k),
        atom_m[0],
        atom_m[1],
        atom_k,
    )
    mma_permute_order = (3, 4, 1, 5, 2, 0)

    # Move reference scale factors to CPU if needed
    ref_sf_cpu = ref_sf.cpu() if ref_sf.is_cuda else ref_sf
    # Reshape to ref_shape format: [mn, scale_k, l] -> [l, mn, scale_k]
    ref_f32_torch_tensor_cpu = ref_sf_cpu.permute(2, 0, 1).contiguous()

    # Create f32 MMA tensor on CPU using PyTorch
    cute_f32_torch_tensor_cpu = torch.randint(0, 1, mma_shape, dtype=torch.float32)
    # Permute to MMA layout
    cute_f32_torch_tensor_cpu = cute_f32_torch_tensor_cpu.permute(mma_permute_order).contiguous()

    # Convert reference f32 tensor to CuTe f32 tensor using layout conversion
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        from_dlpack(ref_f32_torch_tensor_cpu),
        from_dlpack(cute_f32_torch_tensor_cpu),
    )
    # Move to GPU
    cute_f32_torch_tensor = cute_f32_torch_tensor_cpu.cuda()

    # Create CuTe tensor with target dtype (FP8)
    cute_tensor, cute_torch_tensor = cutlass_torch.cute_tensor_like(
        cute_f32_torch_tensor_cpu,
        dtype,
        is_dynamic_layout=True,
        assumed_align=16,
    )

    # Convert f32 CuTe tensor to target dtype CuTe tensor
    cute_tensor = cutlass_torch.convert_cute_tensor(
        cute_f32_torch_tensor,
        cute_tensor,
        dtype,
        is_dynamic_layout=True,
    )
    return cute_tensor, cute_torch_tensor


def custom_kernel(data: input_t) -> output_t:
    """
    Execute the block-scaled GEMV kernel.
    
    This is the main entry point called by the evaluation framework.
    It converts PyTorch tensors to CuTe tensors, launches the kernel,
    and returns the result.
    
    Args:
        data: Tuple of (a, b, scale_a, scale_b, c) PyTorch tensors
            a: [m, k, l] - Input matrix in FP4 (simulated with uint8)
            b: [1, k, l] - Input vector in FP4 (simulated with uint8)
            scale_a: [m, k, l] - Expanded scale factors for a in FP32
            scale_b: [1, k, l] - Expanded scale factors for b in FP32
            c: [m, 1, l] - Output vector in FP16
    
    Returns:
        Output tensor c with computed GEMV results
    """
    a, b, scale_a, scale_b, c = data
    
    # Get dimensions from MxKxL layout
    m, k, l = a.shape
    n = 1  # GEMV: N dimension is always 1
    scale_k = ceil_div(k, block_size)

    # GEMV, N must be 1
    assert n == 1, "GEMV requires N=1"

    # Create reference tensors in LxMxK layout (for CuTe compatibility)
    a_ref = a.to(torch.float32).permute(2, 0, 1).contiguous()  # [l, m, k]
    b_ref = b.to(torch.float32).permute(2, 0, 1).contiguous()  # [l, 1, k]
    c_ref = torch.zeros(l, m, n, dtype=torch.float32, device=a.device)  # [l, m, 1]

    # Create CuTe tensors for A, B, C
    a_tensor, a_torch = cutlass_torch.cute_tensor_like(
        a_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    b_tensor, b_torch = cutlass_torch.cute_tensor_like(
        b_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    c_tensor, c_torch = cutlass_torch.cute_tensor_like(
        c_ref, c_dtype, is_dynamic_layout=True, assumed_align=16
    )

    # Mark tensors with element divisibility for 16B alignment
    a_tensor.mark_compact_shape_dynamic(
        mode=1,
        stride_order=(2, 0, 1),
        divisibility=32,
    )
    b_tensor.mark_compact_shape_dynamic(
        mode=1,
        stride_order=(2, 0, 1),
        divisibility=32,
    )
    c_tensor.mark_compact_shape_dynamic(
        0,
        (2, 1, 0),
        divisibility=16,
    )

    # Extract compact scale factors from expanded scales
    # scale_a and scale_b are [m/n, k, l], we need [m/n, scale_k, l]
    # Take every block_size-th element along k dimension
    scale_a_compact = scale_a[:, ::block_size, :].contiguous()  # [m, scale_k, l]
    scale_b_compact = scale_b[:, ::block_size, :].contiguous()  # [1, scale_k, l]

    # Create scale factor tensors in MMA layout
    sfa_ref, sfa_tensor, sfa_torch = create_scale_factor_tensor(
        l, m, k, block_size, sf_dtype, scale_a_compact
    )
    sfb_ref, sfb_tensor, sfb_torch = create_scale_factor_tensor(
        l, n, k, block_size, sf_dtype, scale_b_compact
    )
    # Run the compiled kernel
    my_kernel(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor)
    return c
