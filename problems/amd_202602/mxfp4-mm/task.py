"""
quant + FP4 GEMM: bf16 A, B -> MXFP4 1x32 per-block quant -> gemm_a4w4 -> bf16 C.
"""
import torch
from typing import TypeVar, TypedDict

# Input: (A, B, B_q, B_shuffle, B_scale_sh) from generate_input.
# A [m,k], B [n,k] bf16; B_q quantized MXFP4; B_shuffle = shuffle_weight(B_q,(16,16)); B_scale_sh from quant(B, shuffle=True).
# Output: bf16 C [m, n].
input_t = TypeVar(
    "input_t",
    bound=tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
)
output_t = TypeVar("output_t", bound=torch.Tensor)


class TestSpec(TypedDict):
    m: int
    n: int
    k: int
    seed: int
