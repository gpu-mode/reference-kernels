#!POPCORN leaderboard vectoradd_py

import torch
import triton
import triton.language as tl
from task import input_t, output_t


@triton.jit
def add_kernel(
    A_ptr, B_ptr, C_ptr, M, N,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask_row = row_idx < M
    mask_col = col_idx < N

    A = tl.load(A_ptr + row_idx[:, None] * N + col_idx[None, :], mask=mask_row[:, None] & mask_col[None, :], other=0.0)
    B = tl.load(B_ptr + row_idx[:, None] * N + col_idx[None, :], mask=mask_row[:, None] & mask_col[None, :], other=0.0)

    C = A + B
    tl.store(C_ptr + row_idx[:, None] * N + col_idx[None, :], C, mask=mask_row[:, None] & mask_col[None, :])

def custom_kernel(data: input_t) -> output_t:
    A, B = data
    M, N = A.shape

    C = torch.empty_like(A)

    BLOCK_SIZE = 32
    grid = (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))

    add_kernel[grid](
        A, B, C, M, N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return C
