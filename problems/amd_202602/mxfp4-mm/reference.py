"""
FP4 quant + FP4 GEMM reference: bf16 A, MXFP4 B -> MXFP4 per-1x32 quant A -> gemm_a4w4 -> bf16 C.
Quant logic follows aiter op_tests/test_gemm_a4w4.py (get_triton_quant(QuantType.per_1x32)).
"""
import torch
from task import input_t, output_t
from utils import make_match_reference

# K must be divisible by 64 (scale group 32 and fp4 pack 2)
SCALE_GROUP_SIZE = 32

def shuffle_weights(x: torch.Tensor, layout=(16, 16)) -> torch.Tensor:
    """
    Pure PyTorch memory swizzle for weights. 
    Assumes x is already a standard uint8 tensor.
    """
    IN, IK = layout
    BK = IK * 2
    
    # uint8 element_size is 1 byte, so K = 16
    K = 16 
    BN = IN
    
    assert x.shape[-2] % BN == 0, f"{x.shape[-2]} % {BN} == {x.shape[-2] % BN}"
    assert x.shape[-1] % BK == 0, f"{x.shape[-1]} % {BK} == {x.shape[-1] % BK}"

    #ported from https://github.com/ROCm/aiter/blob/main/aiter/ops/shuffle.py
    x_ = x.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K)
    x_ = x_.permute(0, 1, 3, 4, 2, 5)
    x_ = x_.contiguous()
    x_ = x_.view(*x.shape)
    
    return x_

#does shuffle of scaled similar to the end of the triton kernel from here https://github.com/ROCm/aiter/blob/main/aiter/utility/fp4_utils.py to be consistent with aiter data creation for input generation
def shuffle_scales(bs_e8m0: torch.Tensor, M_actual: int, N_actual: int) -> torch.Tensor:
    M_alloc = ((M_actual + 255) // 256) * 256
    scaleM_pad = ((M_actual + 31) // 32) * 32
    scaleN_valid = (N_actual + 31) // 32
    scaleN = ((scaleN_valid + 7) // 8) * 8
    
    bs_e8m0_padded = torch.full((scaleM_pad, scaleN), 127, dtype=torch.uint8, device=bs_e8m0.device)
    bs_e8m0_padded[:M_actual, :scaleN_valid] = bs_e8m0
    
    m = torch.arange(scaleM_pad, device=bs_e8m0.device)[:, None]
    n = torch.arange(scaleN, device=bs_e8m0.device)[None, :]
    
    bs_offs_0 = m // 32
    bs_offs_1 = (m % 32) // 16
    bs_offs_2 = (m % 32) % 16
    
    bs_offs_3 = n // 8
    bs_offs_4 = (n % 8) // 4
    bs_offs_5 = (n % 8) % 4
    
    bs_offs = (
        bs_offs_1
        + bs_offs_4 * 2
        + bs_offs_2 * 4
        + bs_offs_5 * 64
        + bs_offs_3 * 256
        + bs_offs_0 * 32 * scaleN
    )
    
    shuffled_flat = torch.full((M_alloc * scaleN,), 127, dtype=torch.uint8, device=bs_e8m0.device)
    shuffled_flat[bs_offs.flatten()] = bs_e8m0_padded.flatten()
    
    return shuffled_flat.view(M_alloc, scaleN)

def generate_input(m: int, n: int, k: int, seed: int):
    """
    Generate random bf16 inputs A [m, k], B [n, k] and quantized MXFP4 B, 
    shuffled B and B_scale. All natively in PyTorch.
    """
    assert k % 64 == 0, "k must be divisible by 64 (scale group 32 and fp4 pack 2)"
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    
    A = torch.randn((m, k), dtype=torch.bfloat16, device="cuda", generator=gen)
    B = torch.randn((n, k), dtype=torch.bfloat16, device="cuda", generator=gen)
    
    B_q, B_scale = quantize_mxfp4_pure_torch(B)
    B_shuffle = shuffle_weights(B_q, layout=(16, 16))
    B_scale_sh = shuffle_scales(B_scale, M_actual=n, N_actual=k)
    
    return (A, B, B_q, B_shuffle, B_scale_sh)

# type helpers from https://github.com/ROCm/aiter/blob/main/aiter/utility/fp4_utils.py in pytorch

def e8m0_to_f32(scale_e8m0_biased: torch.Tensor) -> torch.Tensor:
    scale_e8m0_biased = scale_e8m0_biased.contiguous().view(torch.uint8)
    zero_case = scale_e8m0_biased == 0
    nan_case = scale_e8m0_biased == 0xFF
    
    scale_f32 = scale_e8m0_biased.to(torch.int32) << 23
    scale_f32[zero_case] = 0x00400000
    scale_f32[nan_case] = 0x7F800001
    return scale_f32.view(torch.float32)

def mxfp4_to_f32(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous().view(torch.uint8)
    x = x.repeat_interleave(2, dim=-1)
    x[..., ::2] = x[..., ::2] & 0xF
    x[..., 1::2] = x[..., 1::2] >> 4
    
    mxfp4_list = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]
    mxfp4_in_f32 = torch.tensor(mxfp4_list, dtype=torch.float32, device=x.device)
    return mxfp4_in_f32[x.long()]

#ported from _dynamic_mxfp4_quant_kernel_asm_layout in https://github.com/ROCm/aiter/blob/main/aiter/utility/fp4_utils.py
def quantize_mxfp4(x: torch.Tensor):
    M, N = x.shape
    x_blocks = x.view(M, N // 32, 32).float()

    amax = x_blocks.abs().max(dim=-1, keepdim=True).values
    amax_i32 = amax.view(torch.int32)
    amax_i32 = (amax_i32 + 2097152) & ~8388607
    amax_rounded = amax_i32.view(torch.float32)

    log_amax = torch.log2(amax_rounded)
    scale_e8m0_unbiased = torch.floor(log_amax) - 2
    scale_e8m0_unbiased = torch.nan_to_num(scale_e8m0_unbiased, neginf=-127.0)
    scale_e8m0_unbiased = torch.clamp(scale_e8m0_unbiased, -127, 127)

    bs_e8m0 = (scale_e8m0_unbiased + 127).to(torch.uint8)
    quant_scale = torch.exp2(-scale_e8m0_unbiased)
    
    qx = x_blocks * quant_scale
    qx_i32 = qx.view(torch.int32)
    
    e = (qx_i32 >> 23) & 0xFF
    m = qx_i32 & 0x7FFFFF
    s_bit = (qx < 0).to(torch.uint8) << 3 

    E8_BIAS, E2_BIAS = 127, 1
    adjusted_exponents = E8_BIAS - (e + 1)
    shift_amount = torch.clamp(adjusted_exponents, min=0, max=31)
    
    denorm_m = (4194304 | (m >> 1)) >> shift_amount
    m = torch.where(e < E8_BIAS, denorm_m, m)

    e = torch.maximum(e, torch.tensor(E8_BIAS - E2_BIAS)) - (E8_BIAS - E2_BIAS)

    e2m1_tmp = torch.minimum((((e << 2) | (m >> 21)) + 1) >> 1, torch.tensor(0x7))
    e2m1_value = s_bit | e2m1_tmp.to(torch.uint8)

    e2m1_value = e2m1_value.view(M, N)
    evens = e2m1_value[:, 0::2]
    odds = e2m1_value[:, 1::2]
    x_fp4 = evens | (odds << 4)

    return x_fp4, bs_e8m0.view(M, N // 32)

def ref_kernel(data: input_t) -> output_t:
    """
    Main reference entry point. Bypasses Aiter shuffles by quantizing the pristine 
    unquantized inputs natively.
    """
    A, B, *_ = data

    A_q, A_scale = quantize_mxfp4(A)
    B_q, B_scale = quantize_mxfp4(B)

    M, _ = A.shape
    N, _ = B.shape

    # Dequantize back to f32 for matmul similar to old reference
    x_f32 = mxfp4_to_f32(A_q)
    x_scales = A_scale.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    x_f32 = x_f32 * e8m0_to_f32(x_scales)

    w_f32 = mxfp4_to_f32(B_q)
    w_scales = B_scale.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    w_f32 = w_f32 * e8m0_to_f32(w_scales)

    return torch.mm(x_f32, w_f32.T).to(A.dtype)[:M, :N]

check_implementation = make_match_reference(ref_kernel, rtol=1e-02, atol=1e-02)
