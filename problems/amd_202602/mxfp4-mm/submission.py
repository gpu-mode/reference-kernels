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

    A, B, B_q, B_shuffle, B_scale_sh = data
    A = A.contiguous()
    B = B.contiguous()
    m, k = A.shape
    n, _ = B.shape

    quant_func = aiter.get_triton_quant(QuantType.per_1x32)
    A_q, A_scale_sh = quant_func(A, shuffle=True)
    out_gemm = aiter.gemm_a4w4(
        A_q,
        B_shuffle,
        A_scale_sh,
        B_scale_sh,
        dtype=dtypes.bf16,
        bpreshuffle=True,
    )
    return out_gemm
