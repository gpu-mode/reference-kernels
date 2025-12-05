from torch._higher_order_ops.torchbind import call_torchbind_fake
import cuda.bindings.driver as cuda

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
# Tile sizes for M, N, K dimensions
mma_tiler_mnk= (128, 128, 256)  
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


#  GPU device kernel
@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b1: cute.CopyAtom,
    mB_nkl1: cute.Tensor,
    tma_atom_b2: cute.CopyAtom,
    mB_nkl2: cute.Tensor,
    tma_atom_sfa: cute.CopyAtom,
    mSFA_mkl: cute.Tensor,
    tma_atom_sfb1: cute.CopyAtom,
    mSFB_nkl1: cute.Tensor,
    tma_atom_sfb2: cute.CopyAtom,
    mSFB_nkl2: cute.Tensor,
    mC_mnl: cute.Tensor,
    a_smem_layout_staged: cute.ComposedLayout,
    b_smem_layout_staged: cute.ComposedLayout,
    sfa_smem_layout_staged: cute.Layout,
    sfb_smem_layout_staged: cute.Layout,
    num_tma_load_bytes: cutlass.Constexpr[int],
    epilogue_op: cutlass.Constexpr = lambda x: x
    * (1.0 / (1.0 + cute.math.exp(-x, fastmath=True))),
):
    """
    GPU device kernel performing the batched GEMM computation.
    """
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    tidx = cute.arch.thread_idx()

    #
    # Setup cta/thread coordinates
    #
    # Coords inside cluster
    bidx, bidy, bidz = cute.arch.block_idx()
    mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)

    # Coords outside cluster
    cta_coord = (bidx, bidy, bidz)
    mma_tile_coord_mnl = (
        cta_coord[0] // cute.size(tiled_mma.thr_id.shape),
        cta_coord[1],
        cta_coord[2],
    )
    # Coord inside cta
    tidx, _, _ = cute.arch.thread_idx()

    #
    # Define shared storage for kernel
    #
    @cute.struct
    class SharedStorage:
        ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stage * 2]
        acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_acc_stage * 2]
        tmem_holding_buf: cutlass.Int32

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    # (MMA, MMA_M, MMA_K, STAGE)
    sA = smem.allocate_tensor(
        element_type=ab_dtype,
        layout=a_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=a_smem_layout_staged.inner,
    )
    # (MMA, MMA_N, MMA_K, STAGE)
    sB1 = smem.allocate_tensor(
        element_type=ab_dtype,
        layout=b_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=b_smem_layout_staged.inner,
    )
    # (MMA, MMA_N, MMA_K, STAGE)
    sB2 = smem.allocate_tensor(
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
    sSFB1 = smem.allocate_tensor(
        element_type=sf_dtype,
        layout=sfb_smem_layout_staged,
        byte_alignment=128,
    )
    # (MMA, MMA_N, MMA_K, STAGE)
    sSFB2 = smem.allocate_tensor(
        element_type=sf_dtype,
        layout=sfb_smem_layout_staged,
        byte_alignment=128,
    )

    #
    # Initialize mainloop ab_pipeline, acc_pipeline and their states
    #
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
        producer_group=ab_pipeline_producer_group,
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
    gB_nkl1 = cute.local_tile(
        mB_nkl1, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    # (bN, bK, RestN, RestK, RestL)
    gB_nkl2 = cute.local_tile(
        mB_nkl2, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    gSFA_mkl = cute.local_tile(
        mSFA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    gSFB_nkl1 = cute.local_tile(
        mSFB_nkl1, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    # (bN, bK, RestN, RestK, RestL)
    gSFB_nkl2 = cute.local_tile(
        mSFB_nkl2, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    # (bM, bN, RestM, RestN, RestL)
    gC_mnl = cute.local_tile(
        mC_mnl, cute.slice_(mma_tiler_mnk, (None, None, 0)), (None, None, None)
    )
    k_tile_cnt = cute.size(gA_mkl, mode=[3])

    #
    # Partition global tensor for TiledMMA_A/B/SFA/SFB/C
    #
    # (MMA, MMA_M, MMA_K, RestK)
    thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
    # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
    tCgA = thr_mma.partition_A(gA_mkl)
    # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
    tCgB1 = thr_mma.partition_B(gB_nkl1)
    # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
    tCgB2 = thr_mma.partition_B(gB_nkl2)
    # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
    tCgSFA = thr_mma.partition_A(gSFA_mkl)
    # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
    tCgSFB1 = thr_mma.partition_B(gSFB_nkl1)
    # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
    tCgSFB2 = thr_mma.partition_B(gSFB_nkl2)
    # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
    tCgC = thr_mma.partition_C(gC_mnl)

    #
    # Partition global/shared tensor for TMA load A/B/SFA/SFB
    #
    # TMA Partition_S/D for A
    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestM, RestK, RestL)
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    # TMA Partition_S/D for B1
    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestN, RestK, RestL)
    tBsB1, tBgB1 = cpasync.tma_partition(
        tma_atom_b1,
        0,
        cute.make_layout(1),
        cute.group_modes(sB1, 0, 3),
        cute.group_modes(tCgB1, 0, 3),
    )
    # TMA Partition_S/D for B2
    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestN, RestK, RestL)
    tBsB2, tBgB2 = cpasync.tma_partition(
        tma_atom_b2,
        0,
        cute.make_layout(1),
        cute.group_modes(sB2, 0, 3),
        cute.group_modes(tCgB2, 0, 3),
    )
    #  TMA Partition_S/D for SFA
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
    # TMA Partition_S/D for SFB1
    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestN, RestK, RestL)
    tBsSFB1, tBgSFB1 = cpasync.tma_partition(
        tma_atom_sfb1,
        0,
        cute.make_layout(1),
        cute.group_modes(sSFB1, 0, 3),
        cute.group_modes(tCgSFB1, 0, 3),
    )
    tBsSFB1 = cute.filter_zeros(tBsSFB1)
    tBgSFB1 = cute.filter_zeros(tBgSFB1)
    # TMA Partition_S/D for SFB2
    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestN, RestK, RestL)
    tBsSFB2, tBgSFB2 = cpasync.tma_partition(
        tma_atom_sfb2,
        0,
        cute.make_layout(1),
        cute.group_modes(sSFB2, 0, 3),
        cute.group_modes(tCgSFB2, 0, 3),
    )
    tBsSFB2 = cute.filter_zeros(tBsSFB2)
    tBgSFB2 = cute.filter_zeros(tBgSFB2)

    #
    # Partition shared/tensor memory tensor for TiledMMA_A/B/C
    #
    # (MMA, MMA_M, MMA_K, STAGE)
    tCrA = tiled_mma.make_fragment_A(sA)
    # (MMA, MMA_N, MMA_K, STAGE)
    tCrB1 = tiled_mma.make_fragment_B(sB1)
    # (MMA, MMA_N, MMA_K, STAGE)
    tCrB2 = tiled_mma.make_fragment_B(sB2)
    # (MMA, MMA_M, MMA_N)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    # (MMA, MMA_M, MMA_N)
    tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

    #
    # Alloc tensor memory buffer
    # Make ACC1 and ACC2 tmem tensor
    # ACC1 += A @ B1
    # ACC2 += A @ B2
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
    tCtAcc1 = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
    acc_tmem_ptr1 = cute.recast_ptr(
        acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc1),
        dtype=cutlass.Float32,
    )
    tCtAcc2 = cute.make_tensor(acc_tmem_ptr1, tCtAcc_fake.layout)

    #
    # Make SFA/SFB1/SFB2 tmem tensor
    #
    # SFA tmem layout: (MMA, MMA_M, MMA_K)
    tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
    )
    # Get SFA tmem ptr
    sfa_tmem_ptr = cute.recast_ptr(
        acc_tmem_ptr
        + tcgen05.find_tmem_tensor_col_offset(tCtAcc1)
        + tcgen05.find_tmem_tensor_col_offset(tCtAcc2),
        dtype=sf_dtype,
    )
    tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

    # SFB1, SFB2 tmem layout: (MMA, MMA_N, MMA_K)
    tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
    )
    # Get SFB1 tmem ptr
    sfb_tmem_ptr1 = cute.recast_ptr(
        acc_tmem_ptr
        + tcgen05.find_tmem_tensor_col_offset(tCtAcc1)
        + tcgen05.find_tmem_tensor_col_offset(tCtAcc2)
        + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
        dtype=sf_dtype,
    )
    tCtSFB1 = cute.make_tensor(sfb_tmem_ptr1, tCtSFB_layout)
    # Get SFB2 tmem ptr
    sfb_tmem_ptr2 = cute.recast_ptr(
        acc_tmem_ptr
        + tcgen05.find_tmem_tensor_col_offset(tCtAcc1)
        + tcgen05.find_tmem_tensor_col_offset(tCtAcc2)
        + tcgen05.find_tmem_tensor_col_offset(tCtSFA)
        + tcgen05.find_tmem_tensor_col_offset(tCtSFB1),
        dtype=sf_dtype,
    )
    tCtSFB2 = cute.make_tensor(sfb_tmem_ptr2, tCtSFB_layout)

    #
    # Partition for S2T copy of SFA/SFB1/SFB2
    #
    # Make S2T CopyAtom
    copy_atom_s2t = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE),
        sf_dtype,
    )
    # (MMA, MMA_MN, MMA_K, STAGE)
    tCsSFA_compact = cute.filter_zeros(sSFA)
    # (MMA, MMA_MN, MMA_K)
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
    tCsSFB1_compact = cute.filter_zeros(sSFB1)
    # (MMA, MMA_MN, MMA_K)
    tCtSFB1_compact = cute.filter_zeros(tCtSFB1)
    tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB1_compact)
    thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
    tCsSFB1_compact_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB1_compact)
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
    tCsSFB1_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
        tiled_copy_s2t_sfb, tCsSFB1_compact_s2t_
    )
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
    tCtSFB1_compact_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB1_compact)

    # SFB2 S2T copy and partition
    # (MMA, MMA_MN, MMA_K, STAGE)
    tCsSFB2_compact = cute.filter_zeros(sSFB2)
    # (MMA, MMA_MN, MMA_K)
    tCtSFB2_compact = cute.filter_zeros(tCtSFB2)
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
    tCsSFB2_compact_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB2_compact)
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
    tCsSFB2_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
        tiled_copy_s2t_sfb, tCsSFB2_compact_s2t_
    )
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
    tCtSFB2_compact_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB2_compact)

    #
    # Slice to per mma tile index
    #
    # ((atom_v, rest_v), RestK)
    tAgA = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
    # ((atom_v, rest_v), RestK)
    tBgB1 = tBgB1[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
    # ((atom_v, rest_v), RestK)
    tBgB2 = tBgB2[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
    # ((atom_v, rest_v), RestK)
    tAgSFA = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
    # ((atom_v, rest_v), RestK)
    tBgSFB1 = tBgSFB1[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
    # ((atom_v, rest_v), RestK)
    tBgSFB2 = tBgSFB2[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

    #
    # Execute Data copy and Math computation in the k_tile loop
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

            #  TMA load A/B1/B2/SFA/SFB1/SFB2 to shared memory
            cute.copy(
                tma_atom_a,
                tAgA[(None, ab_empty.count)],
                tAsA[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )
            cute.copy(
                tma_atom_b1,
                tBgB1[(None, ab_empty.count)],
                tBsB1[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )
            cute.copy(
                tma_atom_b2,
                tBgB2[(None, ab_empty.count)],
                tBsB2[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )
            cute.copy(
                tma_atom_sfa,
                tAgSFA[(None, ab_empty.count)],
                tAsSFA[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )
            cute.copy(
                tma_atom_sfb1,
                tBgSFB1[(None, ab_empty.count)],
                tBsSFB1[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )
            cute.copy(
                tma_atom_sfb2,
                tBgSFB2[(None, ab_empty.count)],
                tBsSFB2[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )

            # Wait for AB buffer full
            ab_full = ab_consumer.wait_and_advance()

            #  Copy SFA/SFB1/SFB2 to tmem
            s2t_stage_coord = (None, None, None, None, ab_full.index)
            tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
            tCsSFB1_compact_s2t_staged = tCsSFB1_compact_s2t[s2t_stage_coord]
            tCsSFB2_compact_s2t_staged = tCsSFB2_compact_s2t[s2t_stage_coord]
            cute.copy(
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t_staged,
                tCtSFA_compact_s2t,
            )
            cute.copy(
                tiled_copy_s2t_sfb,
                tCsSFB1_compact_s2t_staged,
                tCtSFB1_compact_s2t,
            )
            cute.copy(
                tiled_copy_s2t_sfb,
                tCsSFB2_compact_s2t_staged,
                tCtSFB2_compact_s2t,
            )

            # tCtAcc1 += tCrA * tCrSFA * tCrB1 * tCrSFB1
            # tCtAcc2 += tCrA * tCrSFA * tCrB2 * tCrSFB2
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
                    tCtSFB1[sf_kblock_coord].iterator,
                )
                cute.gemm(
                    tiled_mma,
                    tCtAcc1,
                    tCrA[kblock_coord],
                    tCrB1[kblock_coord],
                    tCtAcc1,
                )

                tiled_mma.set(
                    tcgen05.Field.SFB,
                    tCtSFB2[sf_kblock_coord].iterator,
                )
                cute.gemm(
                    tiled_mma,
                    tCtAcc2,
                    tCrA[kblock_coord],
                    tCrB2[kblock_coord],
                    tCtAcc2,
                )

                # Enable accumulate on tCtAcc1/tCtAcc2 after first kblock
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
    tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc1)
    thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
    # (T2R_M, T2R_N, EPI_M, EPI_M)
    tTR_tAcc1 = thr_copy_t2r.partition_S(tCtAcc1)
    # (T2R_M, T2R_N, EPI_M, EPI_M)
    tTR_tAcc2 = thr_copy_t2r.partition_S(tCtAcc2)
    # (T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
    tTR_gC = thr_copy_t2r.partition_D(tCgC)
    # (T2R_M, T2R_N, EPI_M, EPI_N）
    tTR_rAcc1 = cute.make_rmem_tensor(
        tTR_gC[None, None, None, None, 0, 0, 0].shape, cutlass.Float32
    )
    # (T2R_M, T2R_N, EPI_M, EPI_N）
    tTR_rAcc2 = cute.make_rmem_tensor(
        tTR_gC[None, None, None, None, 0, 0, 0].shape, cutlass.Float32
    )
    # (T2R_M, T2R_N, EPI_M, EPI_N）
    tTR_rC = cute.make_rmem_tensor(
        tTR_gC[None, None, None, None, 0, 0, 0].shape, c_dtype
    )
    # STG Atom
    simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), c_dtype)
    tTR_gC = tTR_gC[(None, None, None, None, *mma_tile_coord_mnl)]

    # Wait for accumulator buffer full
    acc_full = acc_consumer.wait_and_advance()

    # Copy accumulator to register
    cute.copy(tiled_copy_t2r, tTR_tAcc1, tTR_rAcc1)
    cute.copy(tiled_copy_t2r, tTR_tAcc2, tTR_rAcc2)

    # Silu activation on acc1 and multiply with acc2
    acc_vec1 = epilogue_op(tTR_rAcc1.load())
    acc_vec2 = tTR_rAcc2.load()
    acc_vec = acc_vec1 * acc_vec2

    tTR_rC.store(acc_vec.to(c_dtype))
    # Store C to global memory
    cute.copy(simt_atom, tTR_rC, tTR_gC)

    acc_full.release()
    # Deallocate TMEM
    cute.arch.barrier()
    tmem.free(acc_tmem_ptr)
    return


@cute.jit
def my_kernel(
    a_ptr: cute.Pointer,
    b1_ptr: cute.Pointer,
    b2_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb1_ptr: cute.Pointer,
    sfb2_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    problem_size: tuple,
    epilogue_op: cutlass.Constexpr = lambda x: x
    * (1.0 / (1.0 + cute.math.exp(-x, fastmath=True))),
):
    """
    Host-side JIT function to prepare tensors and launch GPU kernel.
    """
    m, n, k, l = problem_size

    # Setup attributes that depend on gemm inputs
    a_tensor = cute.make_tensor(
        a_ptr,
        cute.make_layout(
            (m, cute.assume(k, 32), l),
            stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32)),
        ),
    )
    b_tensor1 = cute.make_tensor(
        b1_ptr,
        cute.make_layout(
            (n, cute.assume(k, 32), l),
            stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32)),
        ),
    )
    b_tensor2 = cute.make_tensor(
        b2_ptr,
        cute.make_layout(
            (n, cute.assume(k, 32), l),
            stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32)),
        ),
    )
    c_tensor = cute.make_tensor(
        c_ptr, cute.make_layout((cute.assume(m, 32), n, l), stride=(n, 1, m * n))
    )
    # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
    # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
    sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
        a_tensor.shape, sf_vec_size
    )
    sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)

    # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
    sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
        b_tensor1.shape, sf_vec_size
    )
    sfb_tensor1 = cute.make_tensor(sfb1_ptr, sfb_layout)
    sfb_tensor2 = cute.make_tensor(sfb2_ptr, sfb_layout)

    mma_op = tcgen05.MmaMXF4NVF4Op(
        sf_dtype,
        (mma_tiler_mnk[0], mma_tiler_mnk[1], mma_inst_shape_k),
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
    )
    tiled_mma = cute.make_tiled_mma(mma_op)

    cluster_layout_vmnk  = cute.tiled_divide(
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
    # B1 and B2 have the same size thus share the same smem layout
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
    # SFB1 and SFB2 have the same size thus share the same smem layout
    sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        num_ab_stage,
    )
    atom_thr_size = cute.size(tiled_mma.thr_id.shape)

    # Setup TMA for A
    a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        a_tensor,
        a_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk .shape,
    )
    # Setup TMA for B1
    b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
    tma_atom_b1, tma_tensor_b1 = cute.nvgpu.make_tiled_tma_atom_B(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        b_tensor1,
        b_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk .shape,
    )
    # Setup TMA for B2
    tma_atom_b2, tma_tensor_b2 = cute.nvgpu.make_tiled_tma_atom_B(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        b_tensor2,
        b_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk .shape,
    )
    # Setup TMA for SFA
    sfa_smem_layout = cute.slice_(
        sfa_smem_layout_staged , (None, None, None, 0)
    )
    tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        sfa_tensor,
        sfa_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk .shape,
        internal_type=cutlass.Int16,
    )
    # Setup TMA for SFB1
    sfb_smem_layout = cute.slice_(
        sfb_smem_layout_staged , (None, None, None, 0)
    )
    tma_atom_sfb1, tma_tensor_sfb1 = cute.nvgpu.make_tiled_tma_atom_B(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        sfb_tensor1,
        sfb_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk .shape,
        internal_type=cutlass.Int16,
    )
    # Setup TMA for SFB2
    tma_atom_sfb2, tma_tensor_sfb2 = cute.nvgpu.make_tiled_tma_atom_B(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        sfb_tensor2,
        sfb_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk .shape,
        internal_type=cutlass.Int16,
    )

    # Compute TMA load bytes
    a_copy_size = cute.size_in_bytes(ab_dtype, a_smem_layout)
    b_copy_size = cute.size_in_bytes(ab_dtype, b_smem_layout)
    sfa_copy_size = cute.size_in_bytes(sf_dtype, sfa_smem_layout)
    sfb_copy_size = cute.size_in_bytes(sf_dtype, sfb_smem_layout)
    num_tma_load_bytes = (
        a_copy_size + b_copy_size * 2 + sfa_copy_size + sfb_copy_size * 2
    ) * atom_thr_size

    # Compute grid size
    grid = (
        cute.ceil_div(c_tensor.shape[0], mma_tiler_mnk[0]),
        cute.ceil_div(c_tensor.shape[1], mma_tiler_mnk[1]),
        c_tensor.shape[2],
    )

    # Launch the kernel.
    kernel(
        # MMA (Matrix Multiply-Accumulate) configuration
        tiled_mma,                  # Tiled MMA object defining NVFP4 GEMM compute pattern
        
        # TMA (Tensor Memory Accelerator) atoms and tensors for shared input matrix A
        tma_atom_a,                 # TMA copy atom defining how to load A from global memory
        tma_tensor_a,               # Tensor descriptor for A matrix (m, k, l) - shared by both GEMMs
        
        # TMA atoms and tensors for first B matrix (B1)
        tma_atom_b1,                # TMA copy atom defining how to load B1 from global memory
        tma_tensor_b1,              # Tensor descriptor for B1 matrix (n, k, l) - first GEMM
        
        # TMA atoms and tensors for second B matrix (B2)
        tma_atom_b2,                # TMA copy atom defining how to load B2 from global memory
        tma_tensor_b2,              # Tensor descriptor for B2 matrix (n, k, l) - second GEMM
        
        # TMA atoms and tensors for scale factor A (shared)
        tma_atom_sfa,               # TMA copy atom for loading scale factors for A
        tma_tensor_sfa,             # Tensor descriptor for SFA (block scale factors for A) - shared
        
        # TMA atoms and tensors for scale factor B1
        tma_atom_sfb1,              # TMA copy atom for loading scale factors for B1
        tma_tensor_sfb1,            # Tensor descriptor for SFB1 (block scale factors for B1)
        
        # TMA atoms and tensors for scale factor B2
        tma_atom_sfb2,              # TMA copy atom for loading scale factors for B2
        tma_tensor_sfb2,            # Tensor descriptor for SFB2 (block scale factors for B2)
        
        # Output tensor C (stores both C1 and C2 results)
        c_tensor,                   # Output tensor where both GEMM results will be stored (m, n, l)
        
        # Shared memory layouts with staging for pipelined execution
        a_smem_layout_staged,       # Staged shared memory layout for A (includes stage dimension)
        b_smem_layout_staged,       # Staged shared memory layout for B1/B2 (includes stage dimension)
        sfa_smem_layout_staged,     # Staged shared memory layout for SFA (includes stage dimension)
        sfb_smem_layout_staged,     # Staged shared memory layout for SFB1/SFB2 (includes stage dimension)
        
        # Pipeline synchronization parameter
        num_tma_load_bytes,         # Total bytes to load per TMA transaction (for barrier setup)
        
        # Epilogue operation
        epilogue_op,                # Epilogue operation to apply to output (e.g., element-wise ops)
    ).launch(
        grid=grid,
        block=[threads_per_cta, 1, 1],
        cluster=(1, 1, 1),
    )
    return


# Global cache for compiled kernel
_compiled_kernel_cache = None
# This function is used to compile the kernel once and cache it and then allow users to 
# run the kernel multiple times to get more accurate timing results.
def compile_kernel():
    """
    Compile the kernel once and cache it.
    This should be called before any timing measurements.
    
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
    b1_ptr = make_ptr(
        ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    b2_ptr = make_ptr(
        ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    c_ptr = make_ptr(
        c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    sfa_ptr = make_ptr(
        sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32
    )
    sfb1_ptr = make_ptr(
        sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32
    )
    sfb2_ptr = make_ptr(
        sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32
    )

    # Compile the kernel
    _compiled_kernel_cache = cute.compile(my_kernel, a_ptr, b1_ptr, b2_ptr, sfa_ptr, sfb1_ptr, sfb2_ptr, c_ptr, (0, 0, 0, 0))
    
    return _compiled_kernel_cache


def custom_kernel(data: input_t) -> output_t:
    """
    Execute the block-scaled dual GEMM kernel with silu activation,
    C = silu(A @ B1) * (A @ B2).
    
    This is the main entry point called by the evaluation framework.
    It converts PyTorch tensors to CuTe tensors, launches the kernel,
    and returns the result.
    
    Args:
        data: Tuple of (a, b1, b2, sfa_cpu, sfb1_cpu, sfb2_cpu, c) PyTorch tensors
            a: [m, k, l] - Input matrix in float4e2m1fn 
            b1: [n, k, l] - Input matrix in float4e2m1fn 
            b2: [n, k, l] - Input matrix in float4e2m1fn 
            sfa_cpu: [m, k, l] - Scale factors in float8_e4m3fn, used by reference implementation
            sfb1_cpu: [n, k, l] - Scale factors in float8_e4m3fn, used by reference implementation
            sfb2_cpu: [n, k, l] - Scale factors in float8_e4m3fn, used by reference implementation
            sfa_permuted: [32, 4, rest_m, 4, rest_k, l] - Scale factors in float8_e4m3fn
            sfb1_permuted: [32, 4, rest_n, 4, rest_k, l] - Scale factors in float8_e4m3fn
            sfb2_permuted: [32, 4, rest_n, 4, rest_k, l] - Scale factors in float8_e4m3fn
            c: [m, n, l] - Output vector in float16
    
    Returns:
        Output tensor c with computed results
    """
    a, b1, b2, _, _, _, sfa_permuted, sfb1_permuted, sfb2_permuted, c = data
    
    # Ensure kernel is compiled (will use cached version if available)
    # To avoid the compilation overhead, we compile the kernel once and cache it.
    compiled_func = compile_kernel()

    # Get dimensions from MxKxL layout
    _, k, _ = a.shape
    m, n, l = c.shape
    # Torch use e2m1_x2 data type, thus k is halved
    k = k * 2 

    # Create CuTe pointers for A/B/C/SFA/SFB via torch tensor data pointer
    a_ptr = make_ptr(
        ab_dtype, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    b1_ptr = make_ptr(
        ab_dtype, b1.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    b2_ptr = make_ptr(
        ab_dtype, b2.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    c_ptr = make_ptr(
        c_dtype, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    sfa_ptr = make_ptr(
        sf_dtype, sfa_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    sfb1_ptr = make_ptr(
        sf_dtype, sfb1_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    sfb2_ptr = make_ptr(
        sf_dtype, sfb2_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )

    # Execute the compiled kernel
    compiled_func(a_ptr, b1_ptr, b2_ptr, sfa_ptr, sfb1_ptr, sfb2_ptr, c_ptr, (m, n, k, l))

    return c