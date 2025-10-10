# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import cuda.bindings.driver as cuda

import torch

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.blockscaled_layout as blockscaled_utils

mma_tiler_mnk = (128, 1, 64)
ab_dtype = cutlass.Float4E2M1FN
sf_dtype = cutlass.Float8E8M0FNU
c_dtype = cutlass.Float16
sf_vec_size = 16

"""
Below code gives a reference for NVFP4 block-scaled GEMV (General Matrix-Vector Multiplication):

Given:
    - A: a matrix of shape (l, m, k), where l is the batch size, m is the number of rows, k is the number of columns. The data type is Float4E2M1FN
    - SFA: a matrix of shape (l, m, k//scaling_factor_vector), where l is the batch size, m is the number of rows, k is the number of columns, and scaling factor vector size means these elements will share the same scaling factor. The data type is Float8E8M0FNU. The layout matches definition here https://docs.nvidia.com/cuda/cublas/index.html?highlight=fp4#d-block-scaling-factors-layout.
    - b: a batched vector of shape (l, k) and the data type is Float4E2M1FN.
    - SFB: a matrix of shape (l, k//scaling_factor_vector, 128) and the data type is Float8E8M0FNU. The layout matches definition here https://docs.nvidia.com/cuda/cublas/index.html?highlight=fp4#d-block-scaling-factors-layout.
    - c: the output batched vector of shape (l, m) and the data type is Float16.

Operation:
    c = A * b

Assumptions:
    - The matrix A is stored in memory such that the k (column) dimension is contiguous
    - The m dimension is a multiple of 128
    - The k dimension is a multiple of 64

"""


class Sm100BlockScaledDenseGemvKernel:
    def __init__(self):
        self.threads_per_cta = 128

    @cute.jit
    def __call__(
        self,
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        c_tensor: cute.Tensor,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        # (((32, 4), REST_M), ((SF_K, 4), REST_K), (1, REST_L))
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)
        # (((32, 4), REST_M), ((SF_K, 4), REST_K), (1, REST_L))
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_tensor.shape, sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)
        # Compute grid size
        grid = (
            cute.ceil_div(c_tensor.shape[0], 128),
            1,
            c_tensor.shape[2],
        )
        # Launch the kernel synchronously
        self.kernel(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )
        return

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
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

            # Create Tensor for A/B/SFA/SFB tile
            tAgA = cute.make_tensor(tAgA.iterator, mma_tiler_mnk[2])
            tBgB = cute.make_tensor(tBgB.iterator, mma_tiler_mnk[2])
            tAgSFA = cute.make_tensor(tAgSFA.iterator, 4)
            tBgSFB = cute.make_tensor(tBgSFB.iterator, 4)

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

            for i in cutlass.range_constexpr(mma_tiler_mnk[2] // sf_vec_size):
                for j in cutlass.range_constexpr(sf_vec_size):
                    res += (
                        a_val[i * sf_vec_size + j]
                        * sfa_val[i]
                        * b_val[i * sf_vec_size + j]
                        * sfb_val[i]
                    )
        tCgC.store(res.to(cutlass.Float16))
        return


@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_tensor: cute.Tensor,
    sf_mma_tensor: cute.Tensor,
):
    """Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)xK(4xrest_k)xL layout"""
    # sf_mma_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
    # group to ((32, 4, rest_m), (4, rest_k), l)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]


def run_gemv(
    m: int,
    k: int,
    l: int,
    tolerance: float,
):
    """
    Prepare A/B/SFA/SFB/C tensors, launch GPU kernel, and reference checking.
    """
    print("=" * 60)
    print("Launching Blackwell NVFP4 GEMV Test")
    print("-" * 60)
    print("Input dimensions:")
    print(f"  A: ({l}, {m}, {k}) [l: batch size, m: rows, k: cols]")
    print(f"  b: ({l}, {k}) [l: batch size, k: length]")
    print(f"  c: ({l}, {m}) [l: batch size, m: length]")
    print("Data types:")
    print(f"  A/b dtype: {ab_dtype}")
    print(f"  Scaling factor dtype: {sf_dtype} (vector size: {sf_vec_size})")
    print(f"  Output C dtype: {c_dtype}")
    print(f"Validation tolerance: {tolerance}")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    # GEMV, N must be 1
    n = 1

    # Create tensor A/B/C
    a_ref = cutlass_torch.matrix(l, m, k, False, cutlass.Float32)
    b_ref = cutlass_torch.matrix(l, n, k, False, cutlass.Float32)
    c_ref = cutlass_torch.matrix(l, m, n, True, cutlass.Float32)
    a_tensor, a_torch = cutlass_torch.cute_tensor_like(
        a_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    b_tensor, b_torch = cutlass_torch.cute_tensor_like(
        b_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    c_tensor, c_torch = cutlass_torch.cute_tensor_like(
        c_ref, c_dtype, is_dynamic_layout=True, assumed_align=16
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

    #
    # Helper function to create scale factor tensor SFA/SFB
    # for 1x16 block scaled wise use case and follow the layout requirement
    # defined in https://docs.nvidia.com/cuda/cublas/index.html?highlight=fp4#d-block-scaling-factors-layout
    #
    def create_scale_factor_tensor(l, mn, k, sf_vec_size, dtype):
        def ceil_div(a, b):
            return (a + b - 1) // b

        sf_k = ceil_div(k, sf_vec_size)
        ref_shape = (l, mn, sf_k)

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

        ref_permute_order = (1, 2, 0)
        mma_permute_order = (3, 4, 1, 5, 2, 0)

        # Create f32 ref torch tensor (cpu)
        ref_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
            ref_shape,
            torch.float32,
            permute_order=ref_permute_order,
            init_type=cutlass_torch.TensorInitType.RANDOM,
            init_config=cutlass_torch.RandomInitConfig(
                min_val=1,
                max_val=3,
            ),
        )

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
            from_dlpack(ref_f32_torch_tensor_cpu),
            from_dlpack(cute_f32_torch_tensor_cpu),
        )
        cute_f32_torch_tensor = cute_f32_torch_tensor_cpu.cuda()

        # reshape makes memory contiguous
        ref_f32_torch_tensor_cpu = (
            ref_f32_torch_tensor_cpu.permute(2, 0, 1)
            .unsqueeze(-1)
            .expand(l, mn, sf_k, sf_vec_size)
            .reshape(l, mn, sf_k * sf_vec_size)
            .permute(*ref_permute_order)
        )
        # prune to mkl for reference check.
        ref_f32_torch_tensor_cpu = ref_f32_torch_tensor_cpu[:, :k, :]

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
        return ref_f32_torch_tensor_cpu, cute_tensor, cute_torch_tensor

    sfa_ref, sfa_tensor, sfa_torch = create_scale_factor_tensor(
        l, m, k, sf_vec_size, sf_dtype
    )
    sfb_ref, sfb_tensor, sfb_torch = create_scale_factor_tensor(
        l, 1, k, sf_vec_size, sf_dtype
    )

    # Configure gemv kernel
    gemv = Sm100BlockScaledDenseGemvKernel()
    # Initialize Stream
    current_stream = cutlass_torch.default_stream()
    # Compile gemv kernel
    compiled_gemv = cute.compile(
        gemv,
        a_tensor,
        b_tensor,
        sfa_tensor,
        sfb_tensor,
        c_tensor,
        current_stream,
    )

    # Launch GPU kernel
    compiled_gemv(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor, current_stream)

    # Compute reference result, simulate NVFP4 GEMV via 2 FFMA based elementwise multiplication and 1 FFMA based matmul computations
    res_a = torch.einsum("mkl,mkl->mkl", a_ref, sfa_ref)
    res_b = torch.einsum("nkl,nkl->nkl", b_ref, sfb_ref)
    ref = torch.einsum("mkl,nkl->mnl", res_a, res_b)

    # Convert c back to f32 for comparison.
    c_ref_device = c_ref.cuda()
    cute.testing.convert(
        c_tensor,
        from_dlpack(c_ref_device, assumed_align=16).mark_layout_dynamic(leading_dim=0),
    )
    c_ref = c_ref_device.cpu()
    torch.testing.assert_close(c_ref, ref, atol=tolerance, rtol=1e-02)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Example of Sm100 Dense BlockScaled GEMV."
    )
    parser.add_argument(
        "--m",
        type=int,
        default=512,
        help="m dimensions",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=256,
        help="m dimensions",
    )
    parser.add_argument(
        "--l",
        type=int,
        default=1,
        help="l dimension",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-01, help="Tolerance for validation"
    )
    args = parser.parse_args()

    if args.k % mma_tiler_mnk[2] != 0:
        raise ValueError("K must be a multiple of 64 for this GEMV kernel.")
    if args.m % mma_tiler_mnk[0] != 0:
        raise ValueError("M must be a multiple of 128 for this GEMV kernel.")

    run_gemv(
        args.m,
        args.k,
        args.l,
        args.tolerance,
    )
    print("PASS")
