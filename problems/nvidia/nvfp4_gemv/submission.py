from torch._higher_order_ops.torchbind import call_torchbind_fake
import cuda.bindings.driver as cuda

import torch
from task import input_t, output_t

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
import cutlass.utils.blockscaled_layout as blockscaled_utils

# Kernel configuration parameters
mma_tiler_mnk = (128, 1, 64)  # Tile sizes for M, N, K dimensions
ab_dtype = cutlass.Float4E2M1FN  # FP4 data type for A and B
sf_dtype = cutlass.Float8E4M3FN  # FP8 data type for scale factors
c_dtype = cutlass.Float16  # FP16 output type
sf_vec_size = 16  # Scale factor block size (16 elements share one scale)
threads_per_cta = 128  # Number of threads per CUDA thread block

# Helper function for ceiling division
def ceil_div(a, b):
    return (a + b - 1) // b


# Helper function to reorder the scale factor tensor to match the layout defined in
# https://docs.nvidia.com/cuda/cublas/index.html?highlight=fp4#d-block-scaling-factors-layout
@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_ptr: cute.Pointer,
    sf_mma_ptr: cute.Pointer,
    mn: int,
    sf_k: int,
    l: int,
    mma_shape: tuple,
):
    mma_permute_order = (3, 4, 1, 5, 2, 0)
    permuted_shape = tuple(mma_shape[i] for i in mma_permute_order)
    cute_layout = cute.make_ordered_layout(permuted_shape, order=(2, 1, 4, 0, 3, 5))

    sf_ref_tensor = cute.make_tensor(
        sf_ref_ptr, cute.make_layout((mn, sf_k, l), stride=(sf_k, 1, mn * sf_k))
    )
    sf_mma_tensor = cute.make_tensor(sf_mma_ptr, cute_layout)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]


# The CuTe reference implementation for NVFP4 block-scaled GEMV
@cute.kernel
def kernel(
    mA_mkl: cute.Tensor,
    mB_nkl: cute.Tensor,
    mSFA_mkl: cute.Tensor,
    mSFB_nkl: cute.Tensor,
    mC_mnl: cute.Tensor,
):
    # Get CUDA block and thread indices
    bidx, bidy, bidz = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    # Extract the local tile for input matrix A (shape: [block_M, block_K, rest_M, rest_K, rest_L])
    gA_mkl = cute.local_tile(
        mA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    # Extract the local tile for scale factor tensor for A (same shape as gA_mkl)
    # Here, block_M = (32, 4); block_K = (16, 4)
    gSFA_mkl = cute.local_tile(
        mSFA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    # Extract the local tile for input matrix B (shape: [block_N, block_K, rest_N, rest_K, rest_L])
    gB_nkl = cute.local_tile(
        mB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    # Extract the local tile for scale factor tensor for B (same shape as gB_nkl)
    gSFB_nkl = cute.local_tile(
        mSFB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    # Extract the local tile for output matrix C (shape: [block_M, block_N, rest_M, rest_N, rest_L])
    gC_mnl = cute.local_tile(
        mC_mnl, cute.slice_(mma_tiler_mnk, (None, None, 0)), (None, None, None)
    )

    # Select output element corresponding to this thread and block indices
    tCgC = gC_mnl[tidx, None, bidx, bidy, bidz]
    tCgC = cute.make_tensor(tCgC.iterator, 1)
    res = cute.zeros_like(tCgC, cutlass.Float32)

    # Get the number of k tiles (depth dimension) for the reduction loop
    k_tile_cnt = gA_mkl.layout[3].shape
    for k_tile in range(k_tile_cnt):
        tAgA = gA_mkl[tidx, None, bidx, k_tile, bidz]
        tBgB = gB_nkl[None, None, bidy, k_tile, bidz]
        tAgSFA = gSFA_mkl[tidx, None, bidx, k_tile, bidz]
        tBgSFB = gSFB_nkl[None, None, bidy, k_tile, bidz]

        # Load NVFP4 or FP8 values from global memory
        a_val_nvfp4 = tAgA.load()
        b_val_nvfp4 = tBgB.load()
        sfa_val_fp8 = tAgSFA.load()
        sfb_val_fp8 = tBgSFB.load()

        # Convert loaded values to float32 for computation (FFMA)
        a_val = a_val_nvfp4.to(cutlass.Float32)
        b_val = b_val_nvfp4.to(cutlass.Float32)
        sfa_val = sfa_val_fp8.to(cutlass.Float32)
        sfb_val = sfb_val_fp8.to(cutlass.Float32)

        # Iterate over SF vector tiles and compute the scale&matmul accumulation
        for i in cutlass.range_constexpr(mma_tiler_mnk[2] // sf_vec_size):
            for j in cutlass.range_constexpr(sf_vec_size):
                # Accumulate: (A * scaleA * B * scaleB), where scaling is per-vector
                res += (
                    a_val[i * sf_vec_size + j]
                    * sfa_val[i]
                    * b_val[i * sf_vec_size + j]
                    * sfb_val[i]
                )
    # Store the final float16 result back to global memory
    tCgC.store(res.to(cutlass.Float16))
    return


@cute.jit
def my_kernel(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    problem_size: tuple,
):
    """
    Host-side JIT function to prepare tensors and launch GPU kernel.
    """
    m, _, k, l = problem_size
    # Create CuTe Tensor via pointer and problem size.
    a_tensor = cute.make_tensor(
        a_ptr,
        cute.make_layout(
            (m, cute.assume(k, 32), l),
            stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32)),
        ),
    )
    # We use n=128 to create the torch tensor to do fp4 computation via torch._scaled_mm
    # then copy torch tensor to cute tensor for cute customize kernel computation
    # therefore we need to ensure b_tensor has the right stride with this 128 padded size on n.
    n_padded_128 = 128
    b_tensor = cute.make_tensor(
        b_ptr,
        cute.make_layout(
            (n_padded_128, cute.assume(k, 32), l),
            stride=(cute.assume(k, 32), 1, cute.assume(n_padded_128 * k, 32)),
        ),
    )
    c_tensor = cute.make_tensor(
        c_ptr, cute.make_layout((cute.assume(m, 32), 1, l), stride=(1, 1, m))
    )
    # Convert scale factor tensors to MMA layout
    # The layout matches Tensor Core requirements: (((32, 4), REST_M), ((SF_K, 4), REST_K), (1, REST_L))
    sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
        a_tensor.shape, sf_vec_size
    )
    sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)
    
    sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
        b_tensor.shape, sf_vec_size
    )
    sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)
    
    # Compute grid dimensions
    # Grid is (M_blocks, 1, L) where:
    # - M_blocks = ceil(M / 128) to cover all output rows
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


# Reorder scale factor from (mn, l, sf_k) to (32, 4, rest_m, 4, rest_k, l) layout
def create_reordered_scale_factor_tensor(l, mn, k, ref_f8_tensor):
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
    # Create the reordered scale factor tensor (32, 4, rest_m, 4, rest_k, l) on CPU.
    mma_permute_order = (3, 4, 1, 5, 2, 0)
    # Generate a random int8 tensor, then convert to float8_e4m3fn
    rand_int_tensor = torch.randint(0, 2, mma_shape, dtype=torch.int8)
    reordered_f8_tensor = rand_int_tensor.to(dtype=torch.float8_e4m3fn)
    # Permute according to mma_permute_order
    reordered_f8_tensor = reordered_f8_tensor.permute(*mma_permute_order)

    # Helper function to convert scale factor tensor to CUTE-format scale factor tensor
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        make_ptr(
            cutlass.Float8E4M3FN,
            ref_f8_tensor.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        ),
        make_ptr(
            cutlass.Float8E4M3FN,
            reordered_f8_tensor.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        ),
        mn,
        sf_k,
        l,
        mma_shape,
    )
    return reordered_f8_tensor.cuda()


# Global cache for compiled kernel
_compiled_kernel_cache = None

def compile_kernel():
    """
    Compile the kernel once and cache it.
    This should be called before any timing measurements.
    
    Args:
        a, b, scale_a, scale_b, c: Sample tensors with the expected shapes and types
    
    Returns:
        The compiled kernel function
    """
    global _compiled_kernel_cache
    
    if _compiled_kernel_cache is not None:
        return _compiled_kernel_cache
    

    # Create CuTe pointers for A/B/C/SFA/SFB via torch tensor data pointer
    a_ptr = make_ptr(
        ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    b_ptr = make_ptr(
        ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    c_ptr = make_ptr(
        c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    sfa_ptr = make_ptr(
        sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32
    )
    sfb_ptr = make_ptr(
        sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32
    )

    # Compile the kernel
    _compiled_kernel_cache = cute.compile(my_kernel, a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (0, 0, 0, 0))
    
    return _compiled_kernel_cache


def custom_kernel(data: input_t) -> output_t:
    """
    Execute the block-scaled GEMV kernel.
    
    This is the main entry point called by the evaluation framework.
    It converts PyTorch tensors to CuTe tensors, launches the kernel,
    and returns the result.
    
    Args:
        data: Tuple of (a, b, sfa_cpu, sfb_cpu, c) PyTorch tensors
            a: [m, k, l] - Input matrix in float4e2m1fn 
            b: [1, k, l] - Input vector in float4e2m1fn 
            sfa_cpu: [m, k, l] - Scale factors in float8_e4m3fn 
            sfb_cpu: [1, k, l] - Scale factors in float8_e4m3fn 
            c: [m, 1, l] - Output vector in float16
    
    Returns:
        Output tensor c with computed GEMV results
    """
    a, b, sfa_cpu, sfb_cpu, c = data
    
    # Ensure kernel is compiled (will use cached version if available)
    compiled_func = compile_kernel()
    # Get dimensions from MxKxL layout
    m, k, l = a.shape
    # Torch use e2m1_x2 data type, thus k is halved
    k = k * 2 
    # GEMV N dimension is always 1
    n = 1
    # Scaling factor needs to pad the N size to 128
    n_padded_128 = 128

    # Create the reordered scale factor tensors from the reference scale factor tensors via CuTe function.
    sfa_reordered = create_reordered_scale_factor_tensor(l, m, k, sfa_cpu)
    sfb_reordered = create_reordered_scale_factor_tensor(l, n_padded_128, k, sfb_cpu)

    # Create CuTe pointers for A/B/C/SFA/SFB via torch tensor data pointer
    a_ptr = make_ptr(
        ab_dtype, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    b_ptr = make_ptr(
        ab_dtype, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    c_ptr = make_ptr(
        c_dtype, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    sfa_ptr = make_ptr(
        sf_dtype, sfa_reordered.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    sfb_ptr = make_ptr(
        sf_dtype, sfb_reordered.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )

    # Execute the compiled kernel
    compiled_func(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (m, n, k, l))

    return c
