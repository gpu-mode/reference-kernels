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
from cutlass import Float32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import nvvm, llvm

@dsl_user_op
def atomic_add_fp32(a: float | Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    nvvm.atomicrmw(
        res=T.f32(), op=nvvm.AtomicOpKind.FADD, ptr=gmem_ptr.llvm_ptr, a=Float32(a).ir_value()
    )

# Kernel configuration parameters
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
num_ab_stage = 4
# Total number of columns in tmem
num_tmem_alloc_cols = 512

@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stage * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_acc_stage * 2]
    tmem_holding_buf: cutlass.Int32

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
    mC_mnl: cute.Tensor,
    a_smem_layout_staged: cute.ComposedLayout,
    b_smem_layout_staged: cute.ComposedLayout,
    sfa_smem_layout_staged: cute.Layout,
    sfb_smem_layout_staged: cute.Layout,
    num_tma_load_bytes: cutlass.Constexpr[int],
    split_k_factor: cutlass.Constexpr[int],
):
    """
    GPU device kernel performing the batched GEMM computation.
    """
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    bidx, bidy, bidz = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    #
    # Prefetch tma desc
    #
    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        cpasync.prefetch_descriptor(tma_atom_sfa)
        cpasync.prefetch_descriptor(tma_atom_sfb)

    #
    # Allocate shared storage for kernel
    #
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    # (MMA, MMA_M, MMA_K, STAGE)
    sA = smem.allocate_tensor(
        ab_dtype,
        a_smem_layout_staged.outer,
        128,
        a_smem_layout_staged.inner,
    )
    # (MMA, MMA_N, MMA_K, STAGE)
    sB = smem.allocate_tensor(
        ab_dtype,
        b_smem_layout_staged.outer,
        128,
        b_smem_layout_staged.inner,
    )
    # (MMA, MMA_M, MMA_K, STAGE)
    sSFA = smem.allocate_tensor(
        sf_dtype,
        sfa_smem_layout_staged,
        128,
    )
    # (MMA, MMA_N, MMA_K, STAGE)
    sSFB = smem.allocate_tensor(
        sf_dtype,
        sfb_smem_layout_staged,
        128,
    )

    #
    # Initialize pipelines
    #
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=num_ab_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        tx_count=num_tma_load_bytes,
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
    ).make_participants()

    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=num_acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, threads_per_cta),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    ).make_participants()

    #
    # Partition tensors for MMA and make fragments
    #
    # (BM, BK, restM, restK, restL)
    gA = cute.local_tile(
        mA_mkl,
        cute.select(mma_tiler_mnk, mode=[0, 2]),
        (None, None, None),
    )
    # (BN, BK, restN, restK, restL)
    gB = cute.local_tile(
        mB_nkl,
        cute.select(mma_tiler_mnk, mode=[1, 2]),
        (None, None, None),
    )
    # (BM, BK, restM, restK, restL)
    gSFA = cute.local_tile(
        mSFA_mkl,
        cute.select(mma_tiler_mnk, mode=[0, 2]),
        (None, None, None),
    )
    # (BN, BK, restN, restK, restL)
    gSFB = cute.local_tile(
        mSFB_nkl,
        cute.select(mma_tiler_mnk, mode=[1, 2]),
        (None, None, None),
    )
    # (BM, BN, restM, restN, restL)
    gC = cute.local_tile(
        mC_mnl,
        cute.select(mma_tiler_mnk, mode=[0, 1]),
        (None, None, None),
    )

    #
    # Partition global tensor for TiledMMA_A/B/SFA/SFB/C
    #
    # (MMA, MMA_M, MMA_K, RestK)
    thr_mma = tiled_mma.get_slice(0)
    # (MMA, MMA_M, MMA_N, restM, restK, restL)
    tCgA = thr_mma.partition_A(gA)
    # (MMA, MMA_N, MMA_K, restN, restK, restL)
    tCgB = thr_mma.partition_B(gB)
    # (MMA, MMA_M, MMA_K, restM, restK, restL)
    tCgSFA = thr_mma.partition_A(gSFA)
    # (MMA, MMA_N, MMA_K, restN, restK, restL)
    tCgSFB = thr_mma.partition_B(gSFB)
    # (MMA, MMA_M, MMA_N, restM, restN, restL)
    tCgC = thr_mma.partition_C(gC)

    #
    # Partition global/shared tensor for TMA load A/B/SFA/SFB
    #
    # tAsA: ((atom_v, rest_v), STAGE)
    # tAgA: ((atom_v, rest_v), RestM, RestK, RestL)
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    # tBsB: ((atom_v, rest_v), STAGE)
    # tBgB: ((atom_v, rest_v), RestN, RestK, RestL)
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )
    # tAsSFA: ((atom_v, rest_v), STAGE)
    # tAgSFA: ((atom_v, rest_v), RestM, RestK, RestL)
    tAsSFA, tAgSFA = cpasync.tma_partition(
        tma_atom_sfa,
        0,
        cute.make_layout(1),
        cute.group_modes(sSFA, 0, 3),
        cute.group_modes(tCgSFA, 0, 3),
    )
    tAsSFA = cute.filter_zeros(tAsSFA)
    tAgSFA = cute.filter_zeros(tAgSFA)
    # tBsSFB: ((atom_v, rest_v), STAGE)
    # tBgSFB: ((atom_v, rest_v), RestN, RestK, RestL)
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
    # Partition shared memory tensor for TiledMMA_A/B/C
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

    # Wait for tmem to be allocated
    tmem.wait_for_alloc()
    acc_tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)

    # Create the real tensor residing in tmem
    tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

    #
    # Make SFA/SFB tmem tensor
    #
    sfa_tmem_ptr = cute.recast_ptr(
        acc_tmem_ptr + mma_tiler_mnk[1],
        dtype=sf_dtype,
    )
    # (MMA, MMA_M, MMA_K)
    tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        cute.select(sfa_smem_layout_staged, mode=[0, 1, 2]),
    )
    tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

    sfb_tmem_ptr = cute.recast_ptr(
        acc_tmem_ptr + mma_tiler_mnk[0] + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
        dtype=sf_dtype,
    )
    # (MMA, MMA_N, MMA_K)
    tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        cute.select(sfb_smem_layout_staged, mode=[0, 1, 2]),
    )
    tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

    #
    # Partition for S2T copy of SFA/SFB (TiledMMA)
    #
    # Make S2T CopyAtom
    copy_atom_s2t = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE),
        sf_dtype,
    )

    # SFA
    # (MMA, MMA_M, MMA_K, STAGE)
    # ((((32,4),1),(1,4)),1,4,4)
    tCsSFA_compact = cute.filter_zeros(sSFA)
    # (MMA, MMA_N, MMA_K)
    # ((((32,4),4),(1,4)),1,4)
    tCtSFA_compact = cute.filter_zeros(tCtSFA)
    tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(
        copy_atom_s2t,
        tCtSFA_compact
    )    
    thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_M, MMA_K, STAGE)
    # Copy entire SFA tensor for (128, 256) stage
    # ((((32,4,4),4),1),1,1,4,4)
    tCsSFA_compact_s2t_ = thr_copy_s2t_sfa.partition_S(tCsSFA_compact)
    tCsSFA_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
        copy_atom_s2t,
        tCsSFA_compact_s2t_
    )
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_M, MMA_K)
    # (((32,16,4),1),1,1,4)
    tCtSFA_compact_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)

    # SFB
    # (MMA, MMA_N, MMA_K, STAGE)
    tCsSFB_compact = cute.filter_zeros(sSFB)
    # (MMA, MMA_N, MMA_K)
    tCtSFB_compact = cute.filter_zeros(tCtSFB)
    tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(
        copy_atom_s2t,
        tCtSFB_compact
    )
    thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_N, MMA_K, STAGE)
    tCsSFB_compact_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB_compact)
    tCsSFB_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
        copy_atom_s2t,
        tCsSFB_compact_s2t_
    )
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_N, MMA_K)
    tCtSFB_compact_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB_compact)

    #
    # Slice to per mma tile index
    #
    # Get split_k index from grid z dimension
    # Grid is now (M_tiles, N_tiles, L * split_k)
    split_k_idx = bidz % split_k_factor
    batch_idx = bidz // split_k_factor
    
    # Calculate K tile range for this split
    total_k_tiles = cute.size(gA, mode=[3])
    k_tiles_per_split = cute.ceil_div(total_k_tiles, split_k_factor)
    k_tile_start = split_k_idx * k_tiles_per_split
    k_tile_end = min(k_tile_start + k_tiles_per_split, total_k_tiles)
    k_tile_cnt = k_tile_end - k_tile_start

    # ((atom_v, rest_v), RestK)
    tAgA = tAgA[(None, bidx, None, batch_idx)]
    # ((atom_v, rest_v), RestK)
    tBgB = tBgB[(None, bidy, None, batch_idx)]
    # ((atom_v, rest_v), RestK)
    tAgSFA = tAgSFA[(None, bidx, None, batch_idx)]
    # ((atom_v, rest_v), RestK)
    tBgSFB = tBgSFB[(None, bidy, None, batch_idx)]

    # Prefetch num_ab_stage - 2 of TMA loads
    if warp_idx == 0:
        for stage in cutlass.range(num_ab_stage - 2):
            if stage < k_tile_cnt:
                ab_empty = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, k_tile_start + ab_empty.count)],
                    tAsA[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, k_tile_start + ab_empty.count)],
                    tBsB[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )
                cute.copy(
                    tma_atom_sfa,
                    tAgSFA[(None, k_tile_start + ab_empty.count)],
                    tAsSFA[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )
                cute.copy(
                    tma_atom_sfb,
                    tBgSFB[(None, k_tile_start + ab_empty.count)],
                    tBsSFB[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )

    #
    # Execute Data copy and Math computation in the k_tile loop
    #
    if warp_idx == 0:
        # Reset the ACCUMULATE field for each tile
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        # Wait for accumulator buffer empty
        acc_empty = acc_producer.acquire_and_advance()
        # Execute k_tile loop
        for k_tile in cutlass.range(k_tile_cnt):
            # Issue TMA loads
            if k_tile + num_ab_stage - 2 < k_tile_cnt:
                ab_empty = ab_producer.acquire_and_advance()

                cute.copy(
                    tma_atom_a,
                    tAgA[(None, k_tile_start + ab_empty.count)],
                    tAsA[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, k_tile_start + ab_empty.count)],
                    tBsB[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )
                cute.copy(
                    tma_atom_sfa,
                    tAgSFA[(None, k_tile_start + ab_empty.count)],
                    tAsSFA[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )
                cute.copy(
                    tma_atom_sfb,
                    tBgSFB[(None, k_tile_start + ab_empty.count)],
                    tBsSFB[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )

            # Wait for ab data to be ready
            ab_full = ab_consumer.wait_and_advance()

            # Copy SFA/SFB from smem to tmem
            s2t_stage_coord = (None, None, None, None, ab_full.index)
            cute.copy(
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t[s2t_stage_coord],
                tCtSFA_compact_s2t,
            )
            cute.copy(
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t[s2t_stage_coord],
                tCtSFB_compact_s2t,
            )

            # MMA here
            num_k_blocks = cute.size(tCrA, mode=[2])
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (None, None, k_block_idx, ab_full.index)

                # Set SFA/SFB tensor to tiled_mma
                sf_k_block_coord = (None, None, k_block_idx)
                tiled_mma.set(
                    tcgen05.Field.SFA,
                    tCtSFA[sf_k_block_coord].iterator,
                )
                tiled_mma.set(
                    tcgen05.Field.SFB,
                    tCtSFB[sf_k_block_coord].iterator,
                )

                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    tCrA[k_block_coord],
                    tCrB[k_block_coord],
                    tCtAcc,
                )

                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            
            # Release ab buffer
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
    tTR_rAcc = cute.make_rmem_tensor(
        tTR_gC[None, None, None, None, 0, 0, 0].shape, cutlass.Float32
    )
    # (T2R_M, T2R_N, EPI_M, EPI_N）
    tTR_rC = cute.make_rmem_tensor(
        tTR_gC[None, None, None, None, 0, 0, 0].shape, c_dtype
    )
    # STG Atom
    simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), c_dtype)
    tTR_gC = tTR_gC[(None, None, None, None, bidx, bidy, batch_idx)]

    # Wait for accumulator buffer full
    acc_full = acc_consumer.wait_and_advance()

    # Copy accumulator to register
    cute.copy(tiled_copy_t2r, tTR_tAcc, tTR_rAcc)

    # Split-k: use atomic add
    # Flatten and iterate over elements
    base_ptr = tTR_gC.iterator
    layout = tTR_gC.layout

    for i in cutlass.range(cute.size(tTR_rAcc)):
        val = tTR_rAcc[i]
        offset = layout(i)
        ptr = base_ptr + offset
        atomic_add_fp32(val, ptr)

    acc_full.release()

    # if bidx == 0 and bidy == 0 and tidx == 0:
    #     cute.printf("sSFB: {}", sSFB.shape)
    #     cute.printf("tCsSFB_compact: {}", tCsSFB_compact.shape)
    #     cute.printf("tCtSFB: {}", tCtSFB.shape)
    #     cute.printf("tCtSFB_compact: {}", tCtSFB_compact.shape)
    #     cute.printf("tCsSFB_compact_s2t_: {}", tCsSFB_compact_s2t_.shape)
    #     cute.printf("tCtSFB_compact_s2t: {}", tCtSFB_compact_s2t.shape)
    
    # Deallocate TMEM
    pipeline.sync(barrier_id=1)
    tmem.free(acc_tmem_ptr)

    return


@cute.jit
def my_kernel(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    problem_size: tuple,
    split_k_factor: cutlass.Constexpr[int],
):
    """
    Host-side JIT function to prepare tensors and launch GPU kernel.
    """
    m, n, k, l = problem_size

    # Setup attributes that depend on gemm inputs
    a_tensor = cute.make_tensor(
        a_ptr,
        cute.make_layout(
            (m, cute.assume(k, 256), l),
            stride=(cute.assume(k, 256), 1, cute.assume(m * k, 256)),
        ),
    )
    b_tensor = cute.make_tensor(
        b_ptr,
        cute.make_layout(
            (n, cute.assume(k, 256), l),
            stride=(cute.assume(k, 256), 1, cute.assume(n * k, 256)),
        ),
    )
    c_tensor = cute.make_tensor(
        c_ptr, cute.make_layout((cute.assume(m, 128), n, l), stride=(n, 1, m * n))
    )

    # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
    # (((32,4), Rest_M),((16,4), Rest_K),RestL)
    sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
        a_tensor.shape, sf_vec_size
    )
    sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)

    # (((32,4), Rest_N),((16,4), Rest_K),RestL)
    sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
        b_tensor.shape, sf_vec_size
    )
    sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

    # Setup tiled MMA
    mma_op = tcgen05.MmaMXF4NVF4Op(
        sf_dtype,
        (mma_tiler_mnk[0], mma_tiler_mnk[1], mma_inst_shape_k),
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
    )
    tiled_mma = cute.make_tiled_mma(mma_op)

    # Compute smem layouts for A/B/SFA/SFB
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

    # Setup TMA copy atoms for A/B/SFA/SFB
    tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)

    a_smem_layout = cute.select(a_smem_layout_staged, mode=[0, 1, 2])
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        tma_op,
        a_tensor,
        a_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
    )

    b_smem_layout = cute.select(b_smem_layout_staged, mode=[0, 1, 2])
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        tma_op,
        b_tensor,
        b_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
    )

    sfa_smem_layout = cute.select(sfa_smem_layout_staged, mode=[0, 1, 2])
    sfa_tma_atom, sfa_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        tma_op,
        sfa_tensor,
        sfa_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        internal_type=cutlass.Int16,
    )

    sfb_smem_layout = cute.select(sfb_smem_layout_staged, mode=[0, 1, 2])
    sfb_tma_atom, sfb_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        tma_op,
        sfb_tensor,
        sfb_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        internal_type=cutlass.Int16,
    )

    # cute.printf("Problem size: {}", problem_size)
    # cute.printf("a_smem_layout_staged: {}", a_smem_layout_staged)
    # cute.printf("b_smem_layout_staged: {}", b_smem_layout_staged)
    # cute.printf("sfa_smem_layout_staged: {}", sfa_smem_layout_staged)
    # cute.printf("sfb_smem_layout_staged: {}", sfb_smem_layout_staged)

    a_smem_size = cute.size_in_bytes(ab_dtype, a_smem_layout)
    b_smem_size = cute.size_in_bytes(ab_dtype, b_smem_layout)
    sfa_smem_size = cute.size_in_bytes(sf_dtype, sfa_smem_layout)
    sfb_smem_size = cute.size_in_bytes(sf_dtype, sfb_smem_layout)
    num_tma_load_bytes = a_smem_size + b_smem_size + sfa_smem_size + sfb_smem_size

    # Compute grid size
    grid = (
        cute.ceil_div(c_tensor.shape[0], mma_tiler_mnk[0]),
        cute.ceil_div(c_tensor.shape[1], mma_tiler_mnk[1]),
        c_tensor.shape[2] * split_k_factor,
    )

    # Launch the kernel
    kernel(
        tiled_mma,
        a_tma_atom,
        a_tma_tensor,
        b_tma_atom,
        b_tma_tensor,
        sfa_tma_atom,
        sfa_tma_tensor,
        sfb_tma_atom,
        sfb_tma_tensor,
        c_tensor,
        a_smem_layout_staged,
        b_smem_layout_staged,
        sfa_smem_layout_staged,
        sfb_smem_layout_staged,
        num_tma_load_bytes,
        split_k_factor,
    ).launch(
        grid=grid,
        block=[threads_per_cta, 1, 1],
    )
    
    return


# Global cache for compiled kernel
_compiled_kernel_cache = {}

# This function is used to compile the kernel once and cache it and then allow users to 
# run the kernel multiple times to get more accurate timing results.
def compile_kernel(split_k: int):
    """
    Compile the kernel once and cache it.
    This should be called before any timing measurements.

    Returns:
        The compiled kernel function
    """
    global _compiled_kernel_cache
    
    if split_k in _compiled_kernel_cache:
        return _compiled_kernel_cache[split_k]

    # Create CuTe pointers for A/B/C/SFA/SFB via torch tensor data pointer
    a_ptr = make_ptr(
        ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    b_ptr = make_ptr(
        ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    c_ptr = make_ptr(
        cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    sfa_ptr = make_ptr(
        sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32
    )
    sfb_ptr = make_ptr(
        sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32
    )

    # Compile the kernel
    _compiled_kernel_cache[split_k] = cute.compile(my_kernel, a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (0, 0, 0, 0), split_k)
    
    return _compiled_kernel_cache[split_k]


def custom_kernel(data: input_t) -> output_t:
    """
    Execute the block-scaled GEMM kernel.
    
    This is the main entry point called by the evaluation framework.
    It converts PyTorch tensors to CuTe tensors, launches the kernel,
    and returns the result.
    
    Args:
        data: Tuple of (a, b, sfa_ref, sfb_ref, sfa_permuted, sfb_permuted, c) PyTorch tensors
            a: [m, k, l] - Input matrix in float4e2m1fn 
            b: [n, k, l] - Input vector in float4e2m1fn 
            sfa_ref: [m, k, l] - Scale factors in float8_e4m3fn, used by reference implementation
            sfb_ref: [n, k, l] - Scale factors in float8_e4m3fn, used by reference implementation
            sfa_permuted: [32, 4, rest_m, 4, rest_k, l] - Scale factors in float8_e4m3fn 
            sfb_permuted: [32, 4, rest_n, 4, rest_k, l] - Scale factors in float8_e4m3fn 
            c: [m, n, l] - Output vector in float16
    
    Returns:
        Output tensor c with computed results
    """
    a, b, _, _, sfa_permuted, sfb_permuted, c = data

    # Get dimensions from MxKxL layout
    m, k, l = a.shape
    n, _, _ = b.shape
    # Torch use e2m1_x2 data type, thus k is halved
    k = k * 2 

    # Calculate tile counts
    num_k_tiles = k // mma_tiler_mnk[2]  # K is always a multiple of tile_k
    split_k = 4 if k == 7168 or k == 16384 else 2
    split_k = min(split_k, num_k_tiles) # Ensure split_k doesn't exceed the number of k_tiles

    # Compile kernel for this split_k (uses cache if already compiled)
    compiled_func = compile_kernel(split_k)

    c_fp32 = torch.zeros((m, n, l), dtype=torch.float32, device=c.device)
    c_ptr = make_ptr(
        cutlass.Float32, c_fp32.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )

    # Create CuTe pointers for A/B/C/SFA/SFB via torch tensor data pointer
    a_ptr = make_ptr(
        ab_dtype, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    b_ptr = make_ptr(
        ab_dtype, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    # c_ptr = make_ptr(
    #     c_dtype, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    # )
    sfa_ptr = make_ptr(
        sf_dtype, sfa_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    sfb_ptr = make_ptr(
        sf_dtype, sfb_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )

    # Execute the compiled kernel
    compiled_func(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (m, n, k, l))

    c.copy_(c_fp32.to(dtype=c.dtype))

    return c