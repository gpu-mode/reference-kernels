"""
FP4 quant + FP4 GEMM reference: bf16 A, MXFP4 B -> MXFP4 per-1x32 quant A -> gemm_a4w4 -> bf16 C.
Quant logic follows aiter op_tests/test_gemm_a4w4.py (get_triton_quant(QuantType.per_1x32)).
"""
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """
    Reference: MXFP4 per-1x32 quant on A; B_shuffle, B_scale_sh from generate_input.
    gemm_a4w4 with bpreshuffle=True.
    """
    import aiter
    from aiter import QuantType, dtypes
    from aiter.ops.triton.quant import dynamic_mxfp4_quant 
    from aiter.utility.fp4_utils import e8m0_shuffle

    def _quant_mxfp4(x, shuffle=True):
        x_fp4, bs_e8m0 = dynamic_mxfp4_quant(x)
        if shuffle:
            bs_e8m0 = e8m0_shuffle(bs_e8m0)
        return x_fp4.view(dtypes.fp4x2), bs_e8m0.view(dtypes.fp8_e8m0)
    
    A, B, B_q, B_shuffle, B_scale_sh = data
    A = A.contiguous()
    B = B.contiguous()
    m, k = A.shape
    n, _ = B.shape

    A_q, A_scale_sh = _quant_mxfp4(A, shuffle=True)
    out_gemm = aiter.gemm_a4w4(
        A_q,
        B_shuffle,
        A_scale_sh,
        B_scale_sh,
        dtype=dtypes.bf16,
        bpreshuffle=True,
    )
    return out_gemm
