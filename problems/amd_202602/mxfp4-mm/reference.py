"""
FP4 quant + FP4 GEMM reference: bf16 A, MXFP4 B -> MXFP4 per-1x32 quant A -> gemm_a4w4 -> bf16 C.
Quant logic follows aiter op_tests/test_gemm_a4w4.py (get_triton_quant(QuantType.per_1x32)).

NOTE: Explicitly uses dynamic_mxfp4_quant from aiter.ops.triton.quant (patched in #975)
      rather than going through aiter.get_triton_quant, which may dispatch to the
      unpatched fp4_utils.py kernel. See ROCm/aiter#974, ROCm/aiter#975.
"""
import torch
from task import input_t, output_t
from utils import make_match_reference
from aiter import QuantType,dtypes
import aiter
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.quant import dynamic_mxfp4_quant  # #975-patched kernel
from aiter.utility.fp4_utils import e8m0_shuffle
# K must be divisible by 64 (scale group 32 and fp4 pack 2)
SCALE_GROUP_SIZE = 32

def _quant_mxfp4(x, shuffle=True):
    x_fp4, bs_e8m0 = dynamic_mxfp4_quant(x)
    if shuffle:
        bs_e8m0 = e8m0_shuffle(bs_e8m0)
    return x_fp4.view(dtypes.fp4x2), bs_e8m0.view(dtypes.fp8_e8m0)

def generate_input(m: int, n: int, k: int, seed: int):# -> input_t:
    """
    Generate random bf16 inputs A [m, k], B [n, k] and quantized MXFP4 B, shuffled B and B_scale.

    Returns:
        Tuple of (A, B), both bf16 on cuda.
    """
    assert k % 64 == 0, "k must be divisible by 64 (scale group 32 and fp4 pack 2)"
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    A = torch.randn((m, k), dtype=torch.bfloat16, device="cuda", generator=gen)
    B = torch.randn((n, k), dtype=torch.bfloat16, device="cuda", generator=gen)
    B_q, B_scale_sh = _quant_mxfp4(B, shuffle=True)
    # shuffle B(weight) to (16,16) tile coalesced
    B_shuffle = shuffle_weight(B_q, layout=(16, 16))
    return (A, B, B_q, B_shuffle, B_scale_sh)

def run_torch_fp4_mm(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    PyTorch reference: dequant MXFP4 + E8M0 scale -> f32 -> mm -> dtype.
    Same logic as aiter op_tests/test_gemm_a4w4.run_torch.
    x: [m, k//2] fp4 packed, w: [n, k//2] fp4 packed
    x_scales: [m, k//32] E8M0, w_scales: [n, k//32] E8M0
    Returns: [m, n] in dtype
    """
    from aiter.utility import fp4_utils

    m, _ = x.shape
    n, _ = w.shape
    # fp4 packed -> f32
    x_f32 = fp4_utils.mxfp4_to_f32(x)
    w_f32 = fp4_utils.mxfp4_to_f32(w)
    # E8M0 scale: [*, k//32] -> repeat 32 along k -> f32
    x_scales = x_scales[:m].repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    x_scales_f32 = fp4_utils.e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales[:n].repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    w_scales_f32 = fp4_utils.e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32
    return torch.mm(x_f32, w_f32.T).to(dtype)[:m, :n]


def ref_kernel(data: input_t) -> output_t:
    """
    Reference: MXFP4 per-1x32 quant on A and B; both PyTorch ref and gemm_a4w4 are given.
    Returns gemm_a4w4 for check_implementation.
    """
    A, B, B_q, B_shuffle, B_scale_sh = data
    A = A.contiguous()
    B = B.contiguous()
    m, k = A.shape
    n, _ = B.shape
    
    # 1) PyTorch impl just for your reference: dequant fp4 + e8m0 -> f32 -> mm -> bf16
    # Per-1x32 MXFP4 quant
    # A_q, A_scale = _quant_mxfp4(A, shuffle=False)
    # B_q, B_scale = _quant_mxfp4(B, shuffle=False)

    # gemm_a4w4 expects A [M,K/2], B [N,K/2] as dtypes.fp4x2; A_scale/B_scale [*,K/32] E8M0
    # quant_func returns scale as dtypes.fp8_e8m0; gemm_a4w4 accepts E8M0, no view to uint8 needed
    # slice to exact shapes [m,k_scale] / [n,k_scale] (quant may return padded scale)
    
    # k_scale = k // SCALE_GROUP_SIZE
    # A_scale = A_scale[:m, :k_scale].contiguous()
    # B_scale = B_scale[:n, :k_scale].contiguous()
    # out_torch = run_torch_fp4_mm(A_q, B_q, A_scale, B_scale, torch.bfloat16)

    # 2) aiter.gemm_a4w4 path: needs shuffled B_q and shuffled scales (see test_gemm_a4w4.py:102-105)
    A_q, A_scale_sh = _quant_mxfp4(A, shuffle=True)
    # to be noted, aiter also has other a4w4 implements using triton, https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py
    out_gemm = aiter.gemm_a4w4(
        A_q,
        B_shuffle,
        A_scale_sh,
        B_scale_sh,
        dtype=dtypes.bf16,
        bpreshuffle=True,
    )
    return out_gemm

check_implementation = make_match_reference(ref_kernel, rtol=1e-02, atol=1e-02)