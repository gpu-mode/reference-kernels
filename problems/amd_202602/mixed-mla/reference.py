"""
Pure PyTorch reference for MLA (Multi-head Latent Attention) decode.
No external dependencies beyond PyTorch — no aiter compilation needed.

DeepSeek R1 forward_absorb MLA decode:

  ┌──────────────────────────────────────────────────────────────┐
  │  Q: (total_q, 16, 576) bf16    absorbed query               │
  │  KV: (total_kv, 1, 576) bf16   compressed KV buffer         │
  │                                                              │
  │  K = KV[:, :, :576]    full 576 dims for score computation   │
  │  V = KV[:, :, :512]    first 512 dims (kv_lora_rank)        │
  │                                                              │
  │  For each batch element i:                                   │
  │    scores = Q[i] @ K[i]^T * sm_scale     (16, q, kv)        │
  │    attn   = softmax(scores, dim=-1)                          │
  │    out[i] = attn @ V[i]                  (16, q, 512)        │
  └──────────────────────────────────────────────────────────────┘

Three KV cache formats provided for submissions to choose from:
  "bf16":  (total_kv, 1, 576) bfloat16           — highest precision
  "fp8":   (Tensor, Tensor)  fp8 + scalar scale   — per-tensor quantized
  "mxfp4": (Tensor, Tensor)  fp4x2 + e8m0 scale   — block-32 quantized
"""

import torch
import torch.nn.functional as F
from task import input_t, output_t
from utils import make_match_reference

# ---------------------------------------------------------------------------
# DeepSeek R1 MLA constants (forward_absorb path)
# ---------------------------------------------------------------------------
NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM   # 576
V_HEAD_DIM = KV_LORA_RANK                        # 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)

# FP8 dtype: MI355X (gfx950) uses fnuz variant, NVIDIA uses fn
FP8_DTYPE = getattr(torch, "float8_e4m3fnuz", None) or torch.float8_e4m3fn


# ---------------------------------------------------------------------------
# FP8 quantization (per-tensor, sglang style)
# ---------------------------------------------------------------------------

def quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8 = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8, scale.to(torch.float32).reshape(1)


# ---------------------------------------------------------------------------
# MXFP4 quantization (block-32, pure PyTorch)
#
# FP4 E2M1 format: [sign(1) | exp(2) | mantissa(1)]
#   Positive values by nibble: 0=0, 1=0.5, 2=1, 3=1.5, 4=2, 5=3, 6=4, 7=6
#
# Packing: two fp4 values per uint8 byte
#   low nibble  = even-indexed element (dim 0, 2, 4, ...)
#   high nibble = odd-indexed element  (dim 1, 3, 5, ...)
#
# Scale: one E8M0 byte per block of 32 elements
#   value = 2^(byte - 127)
# ---------------------------------------------------------------------------

# Midpoints between adjacent fp4 positive values, used for nearest rounding
#   [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
#   midpoints: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
_FP4_BOUNDARIES = None


def _fp4_boundaries(device):
    global _FP4_BOUNDARIES
    if _FP4_BOUNDARIES is None or _FP4_BOUNDARIES.device != device:
        _FP4_BOUNDARIES = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
            device=device, dtype=torch.float32,
        )
    return _FP4_BOUNDARIES


def quantize_mxfp4(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-32 MXFP4 quantization.

    Args:
        tensor: (B, M, N) bf16, N divisible by 32

    Returns:
        packed: (B, M, N//2) uint8 — two fp4 e2m1 values per byte
        scale:  (B*M, N//32) uint8 — e8m0 block scales
    """
    B, M, N = tensor.shape
    block_size = 32
    num_blocks = N // block_size

    flat = tensor.reshape(-1, N).float()
    rows = flat.shape[0]
    blocked = flat.view(rows, num_blocks, block_size)

    # E8M0 scale per block: 2^exponent where exponent = ceil(log2(amax / 6.0))
    amax = blocked.abs().amax(dim=-1).clamp(min=1e-12)
    exponent = torch.ceil(torch.log2(amax) - 2.584962500721156)  # log2(6)
    scale_byte = (exponent + 127).clamp(0, 254).to(torch.uint8)
    scale_float = torch.pow(2.0, exponent)

    # Normalize to fp4 range and quantize via bucketize
    normalized = (blocked / scale_float.unsqueeze(-1)).reshape(rows, N)
    sign = (normalized < 0).to(torch.uint8)
    nibble = torch.bucketize(normalized.abs(), _fp4_boundaries(tensor.device))
    fp4 = (sign << 3) | nibble.to(torch.uint8)

    # Pack even/odd elements into low/high nibbles
    packed = fp4[:, 0::2] | (fp4[:, 1::2] << 4)
    return packed.view(B, M, N // 2), scale_byte


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------

def generate_input(
    batchsize: int, qseqlen: int, kvseqlen: int, seed: int,
) -> input_t:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    total_q = batchsize * qseqlen
    total_kv = batchsize * kvseqlen

    q = torch.randn(
        (total_q, NUM_HEADS, QK_HEAD_DIM),
        dtype=torch.bfloat16, device="cuda", generator=gen,
    ) * 0.02

    kv_bf16 = torch.randn(
        (total_kv, NUM_KV_HEADS, QK_HEAD_DIM),
        dtype=torch.bfloat16, device="cuda", generator=gen,
    ) * 0.02

    kv_fp8, kv_scale_fp8 = quantize_fp8(kv_bf16)
    kv_mxfp4, kv_scale_mxfp4 = quantize_mxfp4(kv_bf16)

    kv_data = {
        "bf16": kv_bf16,
        "fp8": (kv_fp8, kv_scale_fp8),
        "mxfp4": (kv_mxfp4, kv_scale_mxfp4),
    }

    qo_indptr = torch.arange(0, batchsize + 1, dtype=torch.int32, device="cuda") * qseqlen
    kv_indptr = torch.arange(0, batchsize + 1, dtype=torch.int32, device="cuda") * kvseqlen

    config = {
        "batch_size": batchsize,
        "num_heads": NUM_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "qk_head_dim": QK_HEAD_DIM,
        "kv_lora_rank": KV_LORA_RANK,
        "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
        "v_head_dim": V_HEAD_DIM,
        "q_seq_len": qseqlen,
        "kv_seq_len": kvseqlen,
        "sm_scale": SM_SCALE,
    }

    return (q, kv_data, qo_indptr, kv_indptr, config)


# ---------------------------------------------------------------------------
# Reference kernel: pure PyTorch bf16 attention
# ---------------------------------------------------------------------------

def ref_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    sm_scale = config["sm_scale"]
    kv_buffer = kv_data["bf16"]
    batch_size = qo_indptr.shape[0] - 1
    out_list = []

    for i in range(batch_size):
        q_s, q_e = int(qo_indptr[i].item()), int(qo_indptr[i + 1].item())
        kv_s, kv_e = int(kv_indptr[i].item()), int(kv_indptr[i + 1].item())

        qi = q[q_s:q_e].float().permute(1, 0, 2)   # (16, seq_q, 576)
        ki = kv_buffer[kv_s:kv_e, 0].float()        # (seq_kv, 576)
        vi = kv_buffer[kv_s:kv_e, 0, :KV_LORA_RANK].float()  # (seq_kv, 512)

        scores = torch.matmul(qi * sm_scale, ki.T)   # (16, seq_q, seq_kv)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, vi)                  # (16, seq_q, 512)
        out_list.append(out.permute(1, 0, 2).to(torch.bfloat16))

    return torch.cat(out_list, dim=0)


check_implementation = make_match_reference(ref_kernel, rtol=1e-02, atol=1e-02)
