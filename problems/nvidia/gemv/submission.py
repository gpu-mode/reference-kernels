import cuda.bindings.driver as cuda

import torch
from task import input_t, output_t
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.blockscaled_layout as blockscaled_utils

mma_tiler_mnk = (128, 1, 64)
ab_dtype = cutlass.Float4E2M1FN
sf_dtype = cutlass.Float8E4M3FN
c_dtype = cutlass.Float16
block_size = 16
threads_per_cta = 128

# Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)xK(4xrest_k)xL layout
@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_tensor: cute.Tensor,
    sf_mma_tensor: cute.Tensor,
):
    # sf_mma_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
    # group to ((32, 4, rest_m), (4, rest_k), l)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]

# GPU device kernel
# Using FFMA to simulate NVFP4 block-scaled GEMV computation
@cute.kernel
def kernel(
    mA_mkl: cute.Tensor,
    mB_nkl: cute.Tensor,
    mSFA_mkl: cute.Tensor,
    mSFB_nkl: cute.Tensor,
    mC_mnl: cute.Tensor,
):
    bidx, bidy, bidz = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    # mma_coord_mnk = (bidx, bidy, bidz)

    # (bM, bK, RestM, RestK, RestL)
    gA_mkl = cute.local_tile(
        mA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    # (bM, bK, RestM, RestK, RestL)
    # bM = (32, 4)
    # bK = (16, 4)
    gSFA_mkl = cute.local_tile(
        mSFA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    # (bN, bK, RestN, RestK, RestL)
    gB_nkl = cute.local_tile(
        mB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    # (bN, bK, RestN, RestK, RestL)
    gSFB_nkl = cute.local_tile(
        mSFB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    # (bM, bN, RestM, RestN, RestL)
    gC_mnl = cute.local_tile(
        mC_mnl, cute.slice_(mma_tiler_mnk, (None, None, 0)), (None, None, None)
    )

    tCgC = gC_mnl[tidx, None, bidx, bidy, bidz]
    tCgC = cute.make_tensor(tCgC.iterator, 1)
    res = cute.zeros_like(tCgC, cutlass.Float32)

    k_tile_cnt = gA_mkl.layout[3].shape
    for k_tile in range(k_tile_cnt):
        tAgA = gA_mkl[tidx, None, bidx, k_tile, bidz]
        tBgB = gB_nkl[None, None, bidy, k_tile, bidz]
        tAgSFA = gSFA_mkl[tidx, None, bidx, k_tile, bidz]
        tBgSFB = gSFB_nkl[None, None, bidy, k_tile, bidz]

        # Load A/B/SFA/SFB tile from global memory
        a_val_nvfp4 = tAgA.load()
        b_val_nvfp4 = tBgB.load()
        sfa_val_fp8 = tAgSFA.load()
        sfb_val_fp8 = tBgSFB.load()

        # Convert to f32 for FFMA computation
        a_val = a_val_nvfp4.to(cutlass.Float32)
        b_val = b_val_nvfp4.to(cutlass.Float32)
        sfa_val = sfa_val_fp8.to(cutlass.Float32)
        sfb_val = sfb_val_fp8.to(cutlass.Float32)

        for i in cutlass.range_constexpr(mma_tiler_mnk[2] // block_size):
            for j in cutlass.range_constexpr(block_size):
                res += (
                    a_val[i * block_size + j]
                    * sfa_val[i]
                    * b_val[i * block_size + j]
                    * sfb_val[i]
                )
    tCgC.store(res.to(cutlass.Float16))
    return

@cute.jit
def my_kernel(
    a_tensor: cute.Tensor,
    b_tensor: cute.Tensor, 
    sfa_tensor: cute.Tensor,
    sfb_tensor: cute.Tensor,
    c_tensor: cute.Tensor):
    # (((32, 4), REST_M), ((SF_K, 4), REST_K), (1, REST_L))
    sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
        a_tensor.shape, block_size
    )
    sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)
    # (((32, 4), REST_M), ((SF_K, 4), REST_K), (1, REST_L))
    sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
        b_tensor.shape, block_size
    )
    sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)
    # Compute grid size
    grid = (
        cute.ceil_div(c_tensor.shape[0], 128),
        1,
        c_tensor.shape[2],
    )
    # Launch the kernel synchronously
    kernel(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor).launch(
        grid=grid,
        block=[threads_per_cta, 1, 1],
        cluster=(1, 1, 1)
    )

def ceil_div(a, b):
    return (a + b - 1) // b

def create_scale_factor_tensor(l, mn, sf_k, ref, dtype):

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

    # Create f32 cute torch tensor (cpu)
    cute_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
        mma_shape,
        torch.float32,
        permute_order=mma_permute_order,
        init_type=cutlass_torch.TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(
            min_val=0,
            max_val=1,
        ),
    )

    # convert ref f32 tensor to cute f32 tensor
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        from_dlpack(ref),
        from_dlpack(cute_f32_torch_tensor_cpu),
    )
    cute_f32_torch_tensor = cute_f32_torch_tensor_cpu.cuda()

    # Create dtype cute torch tensor (cpu)
    cute_tensor, cute_torch_tensor = cutlass_torch.cute_tensor_like(
        cute_f32_torch_tensor_cpu,
        dtype,
        is_dynamic_layout=True,
        assumed_align=16,
    )

    # Convert f32 cute tensor to dtype cute tensor
    cute_tensor = cutlass_torch.convert_cute_tensor(
        cute_f32_torch_tensor,
        cute_tensor,
        dtype,
        is_dynamic_layout=True,
    )
    return cute_tensor, cute_torch_tensor


def custom_kernel(data: input_t):
    """
    Execute the kernel. If not already compiled, compile it first.
    
    Args:
        data: Tuple of (a, b, scale_a, scale_b, c) tensors
    
    Returns:
        Output tensor c
    """
    a, b, scale_a, scale_b, c = data
    # Get dimensions from MxKxL layout
    m = a.shape[0]
    k = a.shape[1]
    l = a.shape[2]
    n = b.shape[0]  # should be 1 for GEMV
    
    # Convert torch tensors to CuTe tensors
    a_tensor, a_torch = cutlass_torch.cute_tensor_like(
        a, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    b_tensor, b_torch = cutlass_torch.cute_tensor_like(
        b, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    c_tensor, c_torch = cutlass_torch.cute_tensor_like(
        c, c_dtype, is_dynamic_layout=True, assumed_align=16
    )

    # Mark tensor with element divisibility for 16B alignment
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
    
    # Create scale tensors
    scale_k = ceil_div(k, block_size)
    scale_a_tensor, scale_a_torch = create_scale_factor_tensor(
        l, m, scale_k, scale_a, sf_dtype
    )
    scale_b_tensor, scale_b_torch = create_scale_factor_tensor(
        l, 1, scale_k, scale_b, sf_dtype
    )
    # Run the compiled kernel
    my_kernel(a_tensor, b_tensor, scale_a_tensor, scale_b_tensor, c_tensor)
    return c

