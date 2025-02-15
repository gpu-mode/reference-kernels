import torch
import triton
import triton.language as tl
from task import input_t, output_t


@triton.jit
def custom_kernel_imp(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # custom triton kernel here
    pass


def custom_kernel(data: input_t) -> output_t:
    A, B = data
    M, N = A.shape

    C = torch.empty_like(A)

    BLOCK_SIZE = 32
    grid = (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))

    custom_kernel_imp[grid](
        A,
        B,
        C,
        M,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return C
