"""
Pure PyTorch reference implementation for MLA (Multi-head Latent Attention)
decode.

This version keeps the same task-facing structure as the original reference:
  - generate_input(...)
  - ref_kernel(...)
  - check_implementation

The key difference is that it avoids AITER entirely. It uses the same dynamic
per-tensor FP8 quantization for Q/KV and computes decode attention with regular
PyTorch matmuls after dequantization. On the official decode workload
(`q_seq_len == 1`) this stays numerically very close to the original a8w8
reference while avoiding the AITER build/compile dependency.
"""

import torch

from task import input_t, output_t
from utils import make_match_reference


# ---------------------------------------------------------------------------
# DeepSeek R1 latent MQA constants (forward_absorb path)
# https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/blob/main/config.json
# ---------------------------------------------------------------------------
NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
V_HEAD_DIM = KV_LORA_RANK
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)

BLOCK_SIZE = 32
FP4_MAX = 6.0

# Query dtype for the reference kernel: "fp8" or "bf16"
Q_DTYPE = "fp8"

# KV cache dtype for the reference kernel: "fp8" or "bf16"
KV_DTYPE = "fp8"


def _default_fp8_dtype() -> torch.dtype:
    if torch.version.hip:
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def _get_fp8_dtype(kv_data: dict) -> torch.dtype:
    kv_fp8_entry = kv_data.get("fp8")
    if isinstance(kv_fp8_entry, tuple):
        kv_buffer_fp8 = kv_fp8_entry[0]
        if kv_buffer_fp8.is_floating_point():
            return kv_buffer_fp8.dtype
    return _default_fp8_dtype()


def _mxfp4_scale_dtype() -> torch.dtype:
    if hasattr(torch, "float8_e8m0fnu"):
        return torch.float8_e8m0fnu
    return torch.float32


# ---------------------------------------------------------------------------
# FP8 quantization (sglang style: dynamic per-tensor)
# ---------------------------------------------------------------------------

def quantize_fp8(
    tensor: torch.Tensor,
    fp8_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dynamic per-tensor FP8 quantization.

    Returns:
        (fp8_tensor, scale) where scale is a scalar float32 tensor.
        Dequantize with: fp8_tensor.to(bf16) * scale
    """
    fp8_dtype = fp8_dtype or _default_fp8_dtype()
    finfo = torch.finfo(fp8_dtype)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8_tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(fp8_dtype)
    return fp8_tensor, scale.to(torch.float32).reshape(1)


def dequantize_fp8(
    fp8_tensor: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    return fp8_tensor.to(dtype) * scale.to(device=fp8_tensor.device, dtype=dtype)


# ---------------------------------------------------------------------------
# Pure PyTorch MXFP4 quantization for kv_data["mxfp4"]
# ---------------------------------------------------------------------------

def _nearest_fp4_code(values: torch.Tensor) -> torch.Tensor:
    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=values.device,
        dtype=torch.float32,
    )
    distances = (values.unsqueeze(-1) - levels).abs()
    return distances.argmin(dim=-1).to(torch.uint8)


def quantize_mxfp4(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Block-32 MXFP4 quantization implemented with native PyTorch types.

    The packed data is emitted as uint8 with two FP4 values per byte, matching the
    task's documented "fp4x2 or uint8 fallback" convention. Scales are stored as
    float8_e8m0fnu when available, otherwise float32.
    """
    if tensor.shape[-1] % BLOCK_SIZE != 0:
        raise ValueError("MXFP4 quantization requires the last dimension divisible by 32")

    orig_shape = tensor.shape
    rows = tensor.reshape(-1, orig_shape[-1]).to(torch.float32)
    num_rows, width = rows.shape
    num_blocks = width // BLOCK_SIZE

    blocked = rows.view(num_rows, num_blocks, BLOCK_SIZE)
    block_amax = blocked.abs().amax(dim=-1).clamp(min=1e-12)
    scale = (block_amax / FP4_MAX).clamp(min=1e-12)

    scale_dtype = _mxfp4_scale_dtype()
    scale_q = scale.to(scale_dtype)
    scale_f32 = scale_q.to(torch.float32)

    normalized = blocked / scale_f32.unsqueeze(-1)
    magnitude_code = _nearest_fp4_code(normalized.abs())
    sign_bit = (normalized < 0).to(torch.uint8) << 3
    nibble = sign_bit | magnitude_code

    packed = nibble[..., 0::2] | (nibble[..., 1::2] << 4)
    packed = packed.reshape(orig_shape[0], orig_shape[1], width // 2).contiguous()
    return packed, scale_q.contiguous()


# ---------------------------------------------------------------------------
# Decode helpers
# ---------------------------------------------------------------------------

def _is_uniform(indptr: torch.Tensor, expected_len: int) -> bool:
    if indptr.numel() <= 1:
        return True
    segments = (indptr[1:] - indptr[:-1]).to(torch.int64)
    target = torch.full_like(segments, expected_len)
    return bool(torch.equal(segments, target))


def _materialize_q(q: torch.Tensor, q_scale: torch.Tensor | None) -> torch.Tensor:
    if q_scale is None:
        return q.to(torch.bfloat16)
    return dequantize_fp8(q, q_scale)


def _materialize_kv(kv_buffer: torch.Tensor, kv_scale: torch.Tensor | None) -> torch.Tensor:
    if kv_scale is None:
        return kv_buffer.to(torch.bfloat16)
    return dequantize_fp8(kv_buffer, kv_scale)


def _decode_uniform(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    q_scale: torch.Tensor | None,
    kv_scale: torch.Tensor | None,
    config: dict,
) -> torch.Tensor:
    batch_size = int(config["batch_size"])
    q_seq_len = int(config["q_seq_len"])
    kv_seq_len = int(config["kv_seq_len"])
    num_heads = int(config["num_heads"])
    qk_head_dim = int(config["qk_head_dim"])
    v_head_dim = int(config["v_head_dim"])
    sm_scale = float(config["sm_scale"])

    q_bf16 = _materialize_q(q, q_scale).view(batch_size, q_seq_len, num_heads, qk_head_dim)
    q_bf16 = q_bf16.permute(0, 2, 1, 3).contiguous()

    kv_bf16 = _materialize_kv(kv_buffer, kv_scale).view(batch_size, kv_seq_len, qk_head_dim)
    k = kv_bf16.unsqueeze(1).transpose(-1, -2).contiguous()
    v = kv_bf16[..., :v_head_dim].unsqueeze(1).contiguous()

    scores = torch.matmul(q_bf16, k)
    probs = torch.softmax(scores.to(torch.float32) * sm_scale, dim=-1).to(torch.bfloat16)
    out = torch.matmul(probs, v)

    return out.permute(0, 2, 1, 3).reshape(batch_size * q_seq_len, num_heads, v_head_dim).contiguous()


def _decode_ragged(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    q_scale: torch.Tensor | None,
    kv_scale: torch.Tensor | None,
    config: dict,
) -> torch.Tensor:
    num_heads = int(config["num_heads"])
    qk_head_dim = int(config["qk_head_dim"])
    v_head_dim = int(config["v_head_dim"])
    sm_scale = float(config["sm_scale"])

    q_bf16 = _materialize_q(q, q_scale)
    kv_bf16 = _materialize_kv(kv_buffer, kv_scale)

    outputs = []
    batch_size = qo_indptr.numel() - 1
    for batch_idx in range(batch_size):
        q_start = int(qo_indptr[batch_idx].item())
        q_end = int(qo_indptr[batch_idx + 1].item())
        kv_start = int(kv_indptr[batch_idx].item())
        kv_end = int(kv_indptr[batch_idx + 1].item())

        q_slice = q_bf16[q_start:q_end].view(-1, num_heads, qk_head_dim).permute(1, 0, 2).contiguous()
        kv_slice = kv_bf16[kv_start:kv_end].view(-1, qk_head_dim)
        k = kv_slice.transpose(0, 1).unsqueeze(0).contiguous()
        v = kv_slice[:, :v_head_dim].unsqueeze(0).contiguous()

        scores = torch.matmul(q_slice.unsqueeze(0), k)
        probs = torch.softmax(scores.to(torch.float32) * sm_scale, dim=-1).to(torch.bfloat16)
        out = torch.matmul(probs, v).squeeze(0).permute(1, 0, 2).contiguous()
        outputs.append(out)

    return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# generate_input / ref_kernel / check_implementation
# ---------------------------------------------------------------------------

def generate_input(batchsize: int, qseqlen: int, kvseqlen: int, seed: int) -> input_t:
    """
    Generate absorbed q and compressed kv_buffer for MLA decode.

    Returns all three KV cache formats in kv_data dict:
      kv_data = {
        "bf16":  Tensor               — (total_kv, 1, 576) bfloat16
        "fp8":   (Tensor, Tensor)     — kv_buffer fp8 + scalar scale
        "mxfp4": (Tensor, Tensor)     — kv_buffer fp4x2/uint8 + fp8_e8m0 scale
      }
    """
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    total_q = batchsize * qseqlen
    total_kv = batchsize * kvseqlen
    fp8_dtype = _default_fp8_dtype()

    q = torch.randn(
        (total_q, NUM_HEADS, QK_HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=gen,
    ) * 0.02

    kv_buffer_bf16 = torch.randn(
        (total_kv, NUM_KV_HEADS, QK_HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=gen,
    ) * 0.02

    kv_buffer_fp8, kv_scale_fp8 = quantize_fp8(kv_buffer_bf16, fp8_dtype)
    kv_buffer_mxfp4, kv_scale_mxfp4 = quantize_mxfp4(kv_buffer_bf16)

    kv_data = {
        "bf16": kv_buffer_bf16,
        "fp8": (kv_buffer_fp8, kv_scale_fp8),
        "mxfp4": (kv_buffer_mxfp4, kv_scale_mxfp4),
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


def ref_kernel(data: input_t) -> output_t:
    """Pure PyTorch MLA decode reference with configurable Q/KV precision."""
    q, kv_data, qo_indptr, kv_indptr, config = data
    fp8_dtype = _get_fp8_dtype(kv_data)

    if Q_DTYPE == "fp8":
        q_input, q_scale = quantize_fp8(q, fp8_dtype)
    else:
        q_input, q_scale = q, None

    if KV_DTYPE == "fp8":
        kv_input, kv_scale = kv_data["fp8"]
    else:
        kv_input, kv_scale = kv_data["bf16"], None

    if (
        int(config["num_kv_heads"]) == 1
        and _is_uniform(qo_indptr, int(config["q_seq_len"]))
        and _is_uniform(kv_indptr, int(config["kv_seq_len"]))
    ):
        return _decode_uniform(q_input, kv_input, q_scale, kv_scale, config)

    return _decode_ragged(q_input, kv_input, qo_indptr, kv_indptr, q_scale, kv_scale, config)


check_implementation = make_match_reference(ref_kernel, rtol=1e-02, atol=1e-02)
