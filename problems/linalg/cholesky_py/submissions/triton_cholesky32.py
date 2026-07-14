import torch
import triton
import triton.language as tl

from task import input_t, output_t


@triton.jit
def _cholesky32_left_kernel(input_ptr, output_ptr, matrix_stride: tl.constexpr):
    matrix = tl.program_id(0)
    row_ids = tl.arange(0, 32)
    col_ids = tl.arange(0, 32)
    rows = row_ids[:, None]
    cols = col_ids[None, :]
    offsets = matrix * matrix_stride + rows * 32 + cols
    values = tl.where(rows >= cols, tl.load(input_ptr + offsets), 0.0)

    for k in range(32):
        row = tl.sum(tl.where(rows == k, values, 0.0), axis=0)
        diagonal = tl.sum(tl.where(col_ids == k, row, 0.0), axis=0)
        diagonal -= tl.sum(tl.where(col_ids < k, row * row, 0.0), axis=0)
        diagonal = tl.sqrt(tl.maximum(diagonal, 0.0))

        column = tl.sum(tl.where(cols == k, values, 0.0), axis=1)
        products = tl.where(cols < k, values * row[None, :], 0.0)
        column = (column - tl.sum(products, axis=1)) / diagonal
        values = tl.where((rows == k) & (cols == k), diagonal, values)
        values = tl.where((rows > k) & (cols == k), column[:, None], values)

    tl.store(output_ptr + offsets, values)


def custom_kernel(data: input_t) -> output_t:
    batch, n, _ = data.shape
    if n != 32:
        return torch.linalg.cholesky_ex(data, check_errors=False).L

    output = torch.empty_like(data)
    _cholesky32_left_kernel[(batch,)](
        data,
        output,
        n * n,
        num_warps=1,
    )
    return output
