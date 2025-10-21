from torch._higher_order_ops.torchbind import call_torchbind_fake
import cuda.bindings.driver as cuda
import functools
from typing import Tuple, List

import torch
from task import input_t, output_t

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.torch as cutlass_torch
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import make_ptr

# Kernel configuration parameters

# Size of tma descriptor in bytes
bytes_per_tensormap = 128
# Number of tensormaps: a, b, sfa, sfb
num_tensormaps = 4
# Tile sizes for M, N, K dimensions
mma_tiler_mnk = (128, 128, 256)  
# Shape of the K dimension for the MMA instruction
mma_inst_shape_k = 64
# FP4 data type for A and B
ab_dtype = cutlass.Float4E2M1FN  
# FP8 data type for scale factors
sf_dtype = cutlass.Float8E4M3FN  
# FP16 output type
c_dtype = cutlass.Float16  
# Scale factor block size (16 elements share one scale)
sf_vec_size = 16  
# Number of threads per CUDA thread block
threads_per_cta = 128  
# Stage numbers of shared memory and tmem
num_acc_stage = 1
num_ab_stage = 1
# Total number of columns in tmem
num_tmem_alloc_cols = 512


# Helper function for ceiling division
def ceil_div(a, b):
    return (a + b - 1) // b


# The CuTe reference implementation for NVFP4 block-scaled GEMM
@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    tma_atom_sfa: cute.CopyAtom,
    mSFA_mkl: cute.Tensor,
    tma_atom_sfb: cute.CopyAtom,
    mSFB_nkl: cute.Tensor,
    tensor_of_abc_ptrs: cute.Tensor,
    tensor_of_sfasfb_ptrs: cute.Tensor,
    tensormaps: cute.Tensor,
    tensor_of_problem_sizes: cute.Tensor,
    a_smem_layout_staged: cute.ComposedLayout,
    b_smem_layout_staged: cute.ComposedLayout,
    sfa_smem_layout_staged: cute.Layout,
    sfb_smem_layout_staged: cute.Layout,
    cta_mn_list: List[Tuple[int, int]],
    num_tma_load_bytes: cutlass.Constexpr[int],
):
    """
    GPU device kernel performing the Group GEMM computation.
    """
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    tidx, _, _ = cute.arch.thread_idx()

    #
    # Delinearize bidz to coord_x, coord_y and group_idx for each CTA
    #
    bidx, bidy, bidz = cute.arch.block_idx()
    group_idx = 0
    find = False
    coord_x = 0
    coord_y = 0
    cta_rest = bidz
    for _, (cta_m, cta_n) in enumerate(cta_mn_list):
        if cta_rest >= (cta_m * cta_n):
            group_idx += 1
            cta_rest -= cta_m * cta_n
        else:
            if not find:
                coord_y = cta_rest // cta_m
                coord_x = cta_rest % cta_m
                cta_rest -= cta_m * cta_n
                find = True

    #
    # Construct C Tensor for each CTA
    #
    mC_mnl_iter = cute.make_ptr(
        c_dtype, tensor_of_abc_ptrs[group_idx, 2], cute.AddressSpace.gmem
    ).align(32)
    m = tensor_of_problem_sizes[group_idx, 0]
    n = tensor_of_problem_sizes[group_idx, 1]
    k = tensor_of_problem_sizes[group_idx, 2]
    l = tensor_of_problem_sizes[group_idx, 3]

    mC_mnl_layout = cute.make_layout(
        (
            m,
            n,
            l,
        ),
        stride=(
            cute.assume(n, 32),
            1,
            m * n,
        ),
    )
    mC_mnl = cute.make_tensor(mC_mnl_iter, mC_mnl_layout)
    # Local partition for global C Tensor
    # (bM, bN, RestM, RestN, RestL)
    gC_mnl = cute.local_tile(
        mC_mnl, cute.slice_(mma_tiler_mnk, (None, None, 0)), (None, None, 0)
    )

    #
    # Define shared storage for kernel
    #
    size_tensormap_in_i64 = (
        num_tensormaps * bytes_per_tensormap // 8
    )
    @cute.struct
    class SharedStorage:
        tensormap_buffer: cute.struct.MemRange[
            cutlass.Int64, size_tensormap_in_i64
        ]
        ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stage * 2]
        acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_acc_stage * 2]
        tmem_holding_buf: cutlass.Int32
    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    tensormap_smem_ptr = storage.tensormap_buffer.data_ptr()
    tensormap_a_smem_ptr = tensormap_smem_ptr
    tensormap_b_smem_ptr = (
        tensormap_a_smem_ptr
        + bytes_per_tensormap // 8
    )
    tensormap_sfa_smem_ptr = (
        tensormap_b_smem_ptr
        + bytes_per_tensormap // 8
    )
    tensormap_sfb_smem_ptr = (
        tensormap_sfa_smem_ptr
        + bytes_per_tensormap // 8
    )
    # Setup smem tensor for A, B, SFA, SFB
    # (MMA, MMA_M, MMA_K, STAGE)
    sA = smem.allocate_tensor(
        element_type=ab_dtype,
        layout=a_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=a_smem_layout_staged.inner,
    )
    # (MMA, MMA_N, MMA_K, STAGE)
    sB = smem.allocate_tensor(
        element_type=ab_dtype,
        layout=b_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=b_smem_layout_staged.inner,
    )
    # (MMA, MMA_M, MMA_K, STAGE)
    sSFA = smem.allocate_tensor(
        element_type=sf_dtype,
        layout=sfa_smem_layout_staged,
        byte_alignment=128,
    )
    # (MMA, MMA_N, MMA_K, STAGE)
    sSFB = smem.allocate_tensor(
        element_type=sf_dtype,
        layout=sfb_smem_layout_staged,
        byte_alignment=128,
    )

    # Initialize mainloop ab_pipeline, acc_pipeline and their states
    ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
        num_stages=num_ab_stage,
        producer_group=ab_pipeline_producer_group,
        consumer_group=ab_pipeline_consumer_group,
        tx_count=num_tma_load_bytes,
    ).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
        num_stages=num_acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            threads_per_cta,
        ),
    ).make_participants()

    #
    # Local_tile partition global tensors
    #
    # (bM, bK, RestM, RestK, RestL)
    gA_mkl = cute.local_tile(
        mA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    # (bN, bK, RestN, RestK, RestL)
    gB_nkl = cute.local_tile(
        mB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    # (bM, bK, RestM, RestK, RestL)
    gSFA_mkl = cute.local_tile(
        mSFA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    # (bN, bK, RestN, RestK, RestL)
    gSFB_nkl = cute.local_tile(
        mSFB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )

    #
    # Partition global tensor for TiledMMA_A/B/C
    #
    thr_mma = tiled_mma.get_slice(0)
    # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
    tCgA = thr_mma.partition_A(gA_mkl)
    # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
    tCgB = thr_mma.partition_B(gB_nkl)
    # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
    tCgSFA = thr_mma.partition_A(gSFA_mkl)
    # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
    tCgSFB = thr_mma.partition_B(gSFB_nkl)
    # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
    tCgC = thr_mma.partition_C(gC_mnl)

    # Update tma descriptor with the correct shapes and strides
    tensormap_manager = utils.TensorMapManager(
        utils.TensorMapUpdateMode.SMEM,
        128,
    )
    tensormap_a_gmem_ptr = tensormap_manager.get_tensormap_ptr(
        tensormaps[(bidz, 0, None)].iterator
    )
    tensormap_b_gmem_ptr = tensormap_manager.get_tensormap_ptr(
        tensormaps[(bidz, 1, None)].iterator
    )
    tensormap_sfa_gmem_ptr = tensormap_manager.get_tensormap_ptr(
        tensormaps[(bidz, 2, None)].iterator
    )
    tensormap_sfb_gmem_ptr = tensormap_manager.get_tensormap_ptr(
        tensormaps[(bidz, 3, None)].iterator
    )

    mA_mkl_iter = cute.make_ptr(
        ab_dtype, tensor_of_abc_ptrs[group_idx, 0], cute.AddressSpace.gmem
    ).align(32)
    mB_nkl_iter = cute.make_ptr(
        ab_dtype, tensor_of_abc_ptrs[group_idx, 1], cute.AddressSpace.gmem
    ).align(32)
    sfa_mkl_iter = cute.make_ptr(
        sf_dtype, tensor_of_sfasfb_ptrs[group_idx, 0], cute.AddressSpace.gmem
    ).align(32)
    sfb_nkl_iter = cute.make_ptr(
        sf_dtype, tensor_of_sfasfb_ptrs[group_idx, 1], cute.AddressSpace.gmem
    ).align(32)
    mA_mkl_layout = cute.make_layout(
        (m, k, l),
        stride=(
            cute.assume(k, 32),
            1,
            cute.assume(m * k, 32),
        ),
    )
    mB_nkl_layout = cute.make_layout(
        (n, k, l),
        stride=(
            cute.assume(k, 32),
            1,
            cute.assume(n * k, 32),
        ),
    )
    # SFA, SFB follows specialized layout defined
    # here: TODO add linke
    atom_shape = ((32, 4), (sf_vec_size, 4))
    atom_stride = ((16, 4), (0, 1))
    sfa_layout = cute.tile_to_shape(
        cute.make_layout(atom_shape, stride=atom_stride),
        mA_mkl_layout.shape,
        (2, 1, 3),
    )
    sfb_layout = cute.tile_to_shape(
        cute.make_layout(atom_shape, stride=atom_stride),
        mB_nkl_layout.shape,
        (2, 1, 3),
    )
    real_tensor_a = cute.make_tensor(mA_mkl_iter, mA_mkl_layout)
    real_tensor_b = cute.make_tensor(mB_nkl_iter, mB_nkl_layout)
    real_tensor_sfa = cute.make_tensor(sfa_mkl_iter, sfa_layout)
    real_tensor_sfb = cute.make_tensor(sfb_nkl_iter, sfb_layout)

    # Let warp 0 initialize tensormap
    if warp_idx == 0:
        tensormap_manager.init_tensormap_from_atom(
            tma_atom_a, tensormap_a_smem_ptr, 0
        )
        tensormap_manager.init_tensormap_from_atom(
            tma_atom_b, tensormap_b_smem_ptr, 0
        )
        tensormap_manager.init_tensormap_from_atom(
            tma_atom_sfa, tensormap_sfa_smem_ptr, 0
        )
        tensormap_manager.init_tensormap_from_atom(
            tma_atom_sfb, tensormap_sfb_smem_ptr, 0
        )
        tensormap_manager.update_tensormap(
            (
                real_tensor_a,
                real_tensor_b,
                real_tensor_sfa,
                real_tensor_sfb,
            ),
            (tma_atom_a, tma_atom_b, tma_atom_sfa, tma_atom_sfb),
            (
                tensormap_a_gmem_ptr,
                tensormap_b_gmem_ptr,
                tensormap_sfa_gmem_ptr,
                tensormap_sfb_gmem_ptr,
            ),
            0,  # tma warp id
            (
                tensormap_a_smem_ptr,
                tensormap_b_smem_ptr,
                tensormap_sfa_smem_ptr,
                tensormap_sfb_smem_ptr,
            ),
        )

        tensormap_manager.fence_tensormap_update(tensormap_a_gmem_ptr)
        tensormap_manager.fence_tensormap_update(tensormap_b_gmem_ptr)
        tensormap_manager.fence_tensormap_update(tensormap_sfa_gmem_ptr)
        tensormap_manager.fence_tensormap_update(tensormap_sfb_gmem_ptr)

    cute.arch.barrier()

    #
    # Partition global/shared tensor for TMA load A/B/SFA/SFB
    #
    # TMA load A partition_S/D
    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestM, RestK, RestL)
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    # TMA load B partition_S/D
    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestN, RestK, RestL)
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )

    #  TMALDG_SFA partition_S/D
    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestM, RestK, RestL)
    tAsSFA, tAgSFA = cpasync.tma_partition(
        tma_atom_sfa,
        0,
        cute.make_layout(1),
        cute.group_modes(sSFA, 0, 3),
        cute.group_modes(tCgSFA, 0, 3),
    )
    tAsSFA = cute.filter_zeros(tAsSFA)
    tAgSFA = cute.filter_zeros(tAgSFA)

    # TMALDG_SFB partition_S/D
    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestN, RestK, RestL)
    tBsSFB, tBgSFB = cpasync.tma_partition(
        tma_atom_sfb,
        0,
        cute.make_layout(1),
        cute.group_modes(sSFB, 0, 3),
        cute.group_modes(tCgSFB, 0, 3),
    )
    tBsSFB = cute.filter_zeros(tBsSFB)
    tBgSFB = cute.filter_zeros(tBgSFB)

    #
    # Partition shared/tensor memory tensor for TiledMMA_A/B/C
    #
    # (MMA, MMA_M, MMA_K, STAGE)
    tCrA = tiled_mma.make_fragment_A(sA)
    # (MMA, MMA_N, MMA_K, STAGE)
    tCrB = tiled_mma.make_fragment_B(sB)
    # (MMA, MMA_M, MMA_N)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    # (MMA, MMA_M, MMA_N)
    tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
    #
    # Alloc tensor memory buffer
    #
    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=threads_per_cta,
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
    )
    tmem.allocate(num_tmem_alloc_cols)
    tmem.wait_for_alloc()
    acc_tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
    tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

    #
    # Make SFA/SFB tmem tensor
    #
    # Get SFA tmem ptr
    sfa_tmem_ptr = cute.recast_ptr(
        acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc),
        dtype=sf_dtype,
    )
    # (MMA, MMA_M, MMA_K)
    tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
    )
    tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
    # Get SFB tmem ptr
    sfb_tmem_ptr = cute.recast_ptr(
        acc_tmem_ptr
        + tcgen05.find_tmem_tensor_col_offset(tCtAcc)
        + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
        dtype=sf_dtype,
    )
    # (MMA, MMA_N, MMA_K)
    tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
    )
    tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

    #
    # Partition for S2T copy of SFA/SFB
    #
    # Make S2T CopyAtom
    copy_atom_s2t = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE),
        sf_dtype,
    )
    # (MMA, MMA_MN, MMA_K, STAGE)
    tCsSFA_compact = cute.filter_zeros(sSFA)
    tCtSFA_compact = cute.filter_zeros(tCtSFA)
    tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_compact)
    thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
    tCsSFA_compact_s2t_ = thr_copy_s2t_sfa.partition_S(tCsSFA_compact)
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
    tCsSFA_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
        tiled_copy_s2t_sfa, tCsSFA_compact_s2t_
    )
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
    tCtSFA_compact_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)

    # (MMA, MMA_MN, MMA_K, STAGE)
    tCsSFB_compact = cute.filter_zeros(sSFB)
    # (MMA, MMA_MN, MMA_K)
    tCtSFB_compact = cute.filter_zeros(tCtSFB)
    tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB_compact)
    thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
    tCsSFB_compact_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB_compact)
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
    tCsSFB_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
        tiled_copy_s2t_sfb, tCsSFB_compact_s2t_
    )
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
    tCtSFB_compact_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB_compact)

    # Number of K loops
    k_tile_cnt = cute.ceil_div(real_tensor_a.shape[1], mma_tiler_mnk[2])

    #
    # Slice to per mma tile index
    #
    mma_tile_coord_mnl = (coord_x, coord_y, 0)
    # ((atom_v, rest_v), RestK)
    tAgA = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
    # ((atom_v, rest_v), RestK)
    tBgB = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
    # ((atom_v, rest_v), RestK)
    tAgSFA = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
    # ((atom_v, rest_v), RestK)
    tBgSFB = tBgSFB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

    #
    # Main loop
    #
    if warp_idx == 0:
        # Wait for accumulator buffer empty
        acc_empty = acc_producer.acquire_and_advance()
        # Set ACCUMULATE field to False for the first k_tile iteration
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        # Execute k_tile loop
        for k_tile in range(k_tile_cnt):
            # Wait for AB buffer empty
            ab_empty = ab_producer.acquire_and_advance()

            #  TMALDG A/B/SFA/SFB
            cute.copy(
                tma_atom_a,
                tAgA[(None, k_tile)],
                tAsA[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
                tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                    tensormap_a_gmem_ptr,
                    cute.AddressSpace.generic,
                ),
            )
            cute.copy(
                tma_atom_b,
                tBgB[(None, k_tile)],
                tBsB[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
                tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                    tensormap_b_gmem_ptr,
                    cute.AddressSpace.generic,
                ),
            )
            cute.copy(
                tma_atom_sfa,
                tAgSFA[(None, k_tile)],
                tAsSFA[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
                tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                    tensormap_sfa_gmem_ptr,
                    cute.AddressSpace.generic,
                ),
            )
            cute.copy(
                tma_atom_sfb,
                tBgSFB[(None, k_tile)],
                tBsSFB[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
                tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                    tensormap_sfb_gmem_ptr,
                    cute.AddressSpace.generic,
                ),
            )

            # Wait for AB buffer full
            ab_full = ab_consumer.wait_and_advance()

            #  Copy SFA/SFB to tmem
            s2t_stage_coord = (None, None, None, None, ab_full.index)
            tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
            tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
            cute.copy(
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t_staged,
                tCtSFA_compact_s2t,
            )
            cute.copy(
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t_staged,
                tCtSFB_compact_s2t,
            )

            # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
            num_kblocks = cute.size(tCrA, mode=[2])
            for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                kblock_coord = (
                    None,
                    None,
                    kblock_idx,
                    ab_full.index,
                )

                # Set SFA/SFB tensor to tiled_mma
                sf_kblock_coord = (None, None, kblock_idx)
                tiled_mma.set(
                    tcgen05.Field.SFA,
                    tCtSFA[sf_kblock_coord].iterator,
                )
                tiled_mma.set(
                    tcgen05.Field.SFB,
                    tCtSFB[sf_kblock_coord].iterator,
                )

                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    tCrA[kblock_coord],
                    tCrB[kblock_coord],
                    tCtAcc,
                )
                # Enable accumulate on tCtAcc after first kblock
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # Async arrive AB buffer empty
            ab_full.release()
        acc_empty.commit()

    #
    # Epilogue
    # Partition for epilogue
    #
    op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE)
    copy_atom_t2r = cute.make_copy_atom(op, cutlass.Float32)
    tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc)
    thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
    # (T2R_M, T2R_N, EPI_M, EPI_M)
    tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc)
    # (T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
    tTR_gC = thr_copy_t2r.partition_D(tCgC)
    # (T2R_M, T2R_N, EPI_M, EPI_N）
    tTR_rAcc = cute.make_fragment(
        tTR_gC[None, None, None, None, 0, 0].shape, cutlass.Float32
    )
    # (T2R_M, T2R_N, EPI_M, EPI_N）
    tTR_rC = cute.make_fragment(tTR_gC[None, None, None, None, 0, 0].shape, c_dtype)
    # STG Atom
    simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), c_dtype)
    tTR_gC = tTR_gC[(None, None, None, None, coord_x, coord_y)]

    # Release TMEM allocation lock
    tmem.relinquish_alloc_permit()

    # Wait for accumulator buffer full
    acc_full = acc_consumer.wait_and_advance()

    # Copy accumulator to register
    cute.copy(tiled_copy_t2r, tTR_tAcc, tTR_rAcc)
    acc_vec = tTR_rAcc.load()
    tTR_rC.store(acc_vec.to(c_dtype))
    # Store C to global memory
    cute.copy(simt_atom, tTR_rC, tTR_gC)

    acc_full.release()

    # Deallocate TMEM
    cute.arch.barrier()
    tmem.free(acc_tmem_ptr)
    pass


# Host-side JIT function to prepare tensors and launch GPU kernel.
@cute.jit
def my_kernel(
    initial_abc_ptrs: Tuple[cute.Pointer, cute.Pointer, cute.Pointer],
    initial_sfasfb_ptrs: Tuple[cute.Pointer, cute.Pointer],
    initial_idx: Tuple[int, int, int],
    ptr_of_tensor_of_problem_sizes: cute.Pointer,
    ptr_of_tensor_of_abc_ptrs: cute.Pointer,
    ptr_of_tensor_of_sfasfb_ptrs: cute.Pointer,
    total_num_clusters: cutlass.Constexpr[int],
    problem_sizes: List[
        Tuple[int, int, int, int]
    ],  # Problem sizes for each group
    tensor_of_tensormap,
    num_groups: cutlass.Constexpr[int],
):

    tensor_of_abc_ptrs = cute.make_tensor(
        ptr_of_tensor_of_abc_ptrs, cute.make_layout((num_groups, 3), stride=(3, 1))
    )
    tensor_of_sfasfb_ptrs = cute.make_tensor(
        ptr_of_tensor_of_sfasfb_ptrs, cute.make_layout((num_groups, 2), stride=(2, 1))
    )
    tensor_of_problem_sizes = cute.make_tensor(
        ptr_of_tensor_of_problem_sizes, cute.make_layout((num_groups, 4), stride=(4, 1))
    )

    a_ptr, b_ptr, _ = initial_abc_ptrs
    sfa_ptr, sfb_ptr = initial_sfasfb_ptrs
    min_a_idx, min_b_idx, _ = initial_idx
    min_a_shape = problem_sizes[0]
    min_b_shape = problem_sizes[0]
    for group_idx, shape in enumerate(problem_sizes):
        if group_idx == min_a_idx:
            min_a_shape = shape
        if group_idx == min_b_idx:
            min_b_shape = shape

    initial_a = cute.make_tensor(
        a_ptr,
        cute.make_layout(
            (min_a_shape[0], cute.assume(min_a_shape[2], 32), min_a_shape[3]),
            stride=(
                cute.assume(min_a_shape[2], 32),
                1,
                cute.assume(min_a_shape[0] * min_a_shape[2], 32),
            ),
        ),
    )
    min_b_shape = problem_sizes[0]
    initial_b = cute.make_tensor(
        b_ptr,
        cute.make_layout(
            (min_b_shape[1], cute.assume(min_b_shape[2], 32), min_b_shape[3]),
            stride=(
                cute.assume(min_b_shape[2], 32),
                1,
                cute.assume(min_b_shape[1] * min_b_shape[2], 32),
            ),
        ),
    )

    # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
    # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
    sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
        initial_a.shape, sf_vec_size
    )
    initial_sfa = cute.make_tensor(sfa_ptr, sfa_layout)

    # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
    sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
        initial_b.shape, sf_vec_size
    )
    initial_sfb = cute.make_tensor(sfb_ptr, sfb_layout)

    # Select MMA operation
    mma_op = tcgen05.MmaMXF4NVF4Op(
        sf_dtype,
        (mma_tiler_mnk[0], mma_tiler_mnk[1], mma_inst_shape_k),
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
    )
    tiled_mma = cute.make_tiled_mma(mma_op)

    cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout((1, 1, 1)),
        (tiled_mma.thr_id.shape,),
    )

    # Compute A/B/SFA/SFB/C shared memory layout
    a_smem_layout_staged = sm100_utils.make_smem_layout_a(
        tiled_mma,
        mma_tiler_mnk,
        ab_dtype,
        num_ab_stage,
    )
    b_smem_layout_staged = sm100_utils.make_smem_layout_b(
        tiled_mma,
        mma_tiler_mnk,
        ab_dtype,
        num_ab_stage,
    )
    sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        num_ab_stage,
    )
    sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        num_ab_stage,
    )

    atom_thr_size = cute.size(tiled_mma.thr_id.shape)

    # TMA load for A
    a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        initial_a,
        a_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk.shape,
    )
    # TMA load for B
    b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        initial_b,
        b_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk.shape,
    )

    # TMA load for SFA
    sfa_smem_layout = cute.slice_(
        sfa_smem_layout_staged, (None, None, None, 0)
    )
    tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        initial_sfa,
        sfa_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk.shape,
        internal_type=cutlass.Int16,
    )

    # TMA load for SFB
    sfb_smem_layout = cute.slice_(
        sfb_smem_layout_staged, (None, None, None, 0)
    )
    tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        initial_sfb,
        sfb_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk.shape,
        internal_type=cutlass.Int16,
    )
    # Compute TMA load bytes
    a_copy_size = cute.size_in_bytes(ab_dtype, a_smem_layout)
    b_copy_size = cute.size_in_bytes(ab_dtype, b_smem_layout)
    sfa_copy_size = cute.size_in_bytes(sf_dtype, sfa_smem_layout)
    sfb_copy_size = cute.size_in_bytes(sf_dtype, sfb_smem_layout)
    num_tma_load_bytes = (
        a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
    ) * atom_thr_size

    # Store CTA shape information for each Group in a List
    cta_mn_list = []
    for group_idx, (m, n, k, l) in enumerate(problem_sizes):
        x, y = cute.ceil_div(problem_sizes[group_idx][:2], mma_tiler_mnk[0:2])
        cta_mn_list.append((x, y))

    # Compute grid size
    grid = (1, 1, total_num_clusters)

    # Launch the kernel synchronously
    kernel(
        tiled_mma,
        tma_atom_a,
        tma_tensor_a,
        tma_atom_b,
        tma_tensor_b,
        tma_atom_sfa,
        tma_tensor_sfa,
        tma_atom_sfb,
        tma_tensor_sfb,
        tensor_of_abc_ptrs,
        tensor_of_sfasfb_ptrs,
        tensor_of_tensormap,
        tensor_of_problem_sizes,
        a_smem_layout_staged,
        b_smem_layout_staged,
        sfa_smem_layout_staged,
        sfb_smem_layout_staged,
        cta_mn_list,
        num_tma_load_bytes,
    ).launch(
        grid=grid,
        block=[threads_per_cta, 1, 1],
        cluster=(1, 1, 1),
    )
    return


_compiled_kernel_cache = None

def compile_kernel():
    pass

def custom_kernel(data: input_t) -> output_t:
    """
    Execute the block-scaled group GEMM kernel.
    
    This is the main entry point called by the evaluation framework.
    It converts PyTorch tensors to CuTe tensors, launches the kernel,
    and returns the result.
    
    Args:
        data: Tuple of (abc_tensors, sfasfb_tensors, problem_sizes) where:
            abc_tensors: list of tuples (a, b, c) where 
                a is torch.Tensor[float4e2m1fn_x2] of shape [m, k // 2, l]
                b is torch.Tensor[float4e2m1fn_x2] of shape [n, k // 2, l]
                c is torch.Tensor[float16] of shape [m, n, l]
            sfasfb_tensors: list of tuples (sfa, sfb) where 
                sfa is torch.Tensor[float8_e4m3fnuz] of shape [m, k // 16, l]
                sfb is torch.Tensor[float8_e4m3fnuz] of shape [n, k // 16, l]
            problem_sizes: list of tuples (m, n, k, l)
            each group has its own a, b, c, sfa, sfb with different m, n, k, l problem sizes
            l should always be 1 for each group.
            list size is the number of groups.
    
    Returns:
        list of c tensors where c is torch.Tensor[float16] of shape [m, n, l] for each group
    """
    abc_tensors, _, sfasfb_reordered_tensors, problem_sizes = data

    # Choose A, B, C, SFA, SFB with the smallest size to create initial tensormaps
    key_size_a = lambda item: item[1][0] * item[1][2]
    key_size_b = lambda item: item[1][1] * item[1][2]
    key_size_c = lambda item: item[1][0] * item[1][1]
    # Find the indices of the groups with the smallest tensor sizes
    min_a_idx, _ = min(enumerate(problem_sizes), key=key_size_a)
    min_b_idx, _ = min(enumerate(problem_sizes), key=key_size_b)
    min_c_idx, _ = min(enumerate(problem_sizes), key=key_size_c)

    abc_ptrs = []
    sfasfb_ptrs = []
    for i, ((a, b, c), (sfa_reordered, sfb_reordered), (m, n, k, l)) in enumerate(zip(abc_tensors, sfasfb_reordered_tensors, problem_sizes)):
        abc_ptrs.append((a.data_ptr(), b.data_ptr(), c.data_ptr()))
        sfasfb_ptrs.append((sfa_reordered.data_ptr(), sfb_reordered.data_ptr()))

    # Pick the tensor with the smallest size to create initial tensormaps
    initial_cute_abc_ptrs = (
        make_ptr(
            ab_dtype,
            abc_tensors[min_a_idx][0].data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            ab_dtype,
            abc_tensors[min_b_idx][1].data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            c_dtype,
            abc_tensors[min_c_idx][2].data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
    )
    initial_cute_sfasfb_ptrs = (
        make_ptr(
            sf_dtype,
            sfasfb_reordered_tensors[min_a_idx][0].data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            sf_dtype,
            sfasfb_reordered_tensors[min_b_idx][1].data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
    )

    # Create torch tensor to store problem sizes
    # layout (num_groups, 4):(4, 1)
    tensor_of_problem_sizes = torch.tensor(
        problem_sizes, dtype=torch.int32, device="cuda"
    )

    # Create torch tensors to store abc_ptrs and sfasfb_ptrs 
    # layout (num_groups,3):(3, 1)
    tensor_of_abc_ptrs = torch.tensor(abc_ptrs, dtype=torch.int64, device="cuda")
    tensor_of_sfasfb_ptrs = torch.tensor(sfasfb_ptrs, dtype=torch.int64, device="cuda")

    # Compute cluster tile shape
    cta_tile_shape_mn = [128, mma_tiler_mnk[1]]
    cluster_tile_shape_mn = tuple(
        x * y for x, y in zip(cta_tile_shape_mn, (1, 1))
    )
    # Compute total number of cluster tiles we need to compute for given grouped GEMM problem
    total_num_clusters = 0
    num_groups = len(problem_sizes)
    for m, n, _, _ in problem_sizes:
        num_clusters_mn = tuple(
            (x + y - 1) // y for x, y in zip((m, n), cluster_tile_shape_mn)
        )
        total_num_clusters += functools.reduce(lambda x, y: x * y, num_clusters_mn)

    # Preserved buffers for each cluster to update its tma descriptor in device memory
    tensormap_shape = (
        total_num_clusters,
        num_tensormaps,
        bytes_per_tensormap // 8,
    )
    tensor_of_tensormap = torch.empty(tensormap_shape, dtype=torch.int64, device="cuda")

    cute_ptr_of_tensor_of_abc_ptrs = make_ptr(
        cutlass.Int64,
        tensor_of_abc_ptrs.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    cute_ptr_of_tensor_of_sfasfb_ptrs = make_ptr(
        cutlass.Int64,
        tensor_of_sfasfb_ptrs.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    cute_ptr_of_tensor_of_problem_sizes = make_ptr(
        cutlass.Int32,
        tensor_of_problem_sizes.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )

    # Execute the compiled kernel
    my_kernel(
        initial_cute_abc_ptrs,
        initial_cute_sfasfb_ptrs,
        (min_a_idx, min_b_idx, min_c_idx),
        cute_ptr_of_tensor_of_problem_sizes,
        cute_ptr_of_tensor_of_abc_ptrs,
        cute_ptr_of_tensor_of_sfasfb_ptrs,
        total_num_clusters,
        problem_sizes,
        tensor_of_tensormap,
        num_groups,
    )

    res = []
    for i in range(num_groups):
        res.append(abc_tensors[i][2])
    return res