# MLA (Multi-head Latent Attention) Decode Kernel

## Description

Implement a custom MLA attention decode kernel optimized for MI355X.

This is the **inner attention kernel** from DeepSeek R1's `forward_absorb` MLA path.
The absorbed query and compressed KV cache are provided directly — you implement the
attention computation with variable-length batching.

The reference uses **aiter MLA a8w8 decode kernel** (`mla_decode_fwd`, fp8 Q + fp8 KV,
persistent mode). On MI355X, a8w8 is ~2-3x faster than bf16 with negligible accuracy loss.
The reference quantizes Q to fp8 on-the-fly and uses pre-quantized fp8 KV from `kv_data["fp8"]`.

## DeepSeek R1 Forward-Absorb MLA Config

| Parameter | Value | Notes |
|---|---|---|
| num_heads | 16 | Query heads (after TP split) |
| num_kv_heads | 1 | Single shared latent KV head |
| kv_lora_rank | 512 | Latent dimension |
| qk_rope_head_dim | 64 | RoPE embedding dimension |
| qk_head_dim | 576 | kv_lora_rank + qk_rope_head_dim (absorbed q/k dim) |
| v_head_dim | 512 | = kv_lora_rank (output dim) |
| sm_scale | 1/sqrt(576) | |
| q dtype | bfloat16 | Input always bf16; reference quantizes to fp8 on-the-fly |
| kv dtype | bf16 / fp8 / mxfp4 | All three provided simultaneously |
| mode | decode | q_seq_len=1, kv_seq_len up to 8k |

## Reference Kernel

The reference (`ref_kernel`) is configurable via two globals in `reference.py`:

| `Q_DTYPE` | `KV_DTYPE` | Aiter kernel dispatched | Description |
|---|---|---|---|
| `"fp8"` (default) | `"fp8"` (default) | `mla_a8w8_qh16_qseqlen1_gqaratio16_ps` | fp8 Q + fp8 KV — fastest |
| `"bf16"` | `"fp8"` | `mla_a16w8_qh16_m16x4_n16x1_coex0_mask1_ps` | bf16 Q + fp8 KV |
| `"bf16"` | `"bf16"` | `mla_a16w16_qh16_m16x4_n16x1_coex0_mask1_ps` | bf16 Q + bf16 KV — highest precision |

**Note**: `Q_DTYPE="fp8"` + `KV_DTYPE="bf16"` is not a valid combination (no a8w16 kernel exists).

### Reference Latency (MI355X)

| Case | a8w8 (us) | a16w16 (us) | a8w8 speedup |
|---|---|---|---|
| bs=4, kv=1k | ~118 | ~162 | 1.4x |
| bs=4, kv=8k | ~113 | ~177 | 1.6x |
| bs=64, kv=8k | ~171 | ~353 | 2.1x |
| bs=256, kv=8k | ~349 | ~814 | 2.3x |

## KV Buffer Format (forward_absorb)

The compressed KV buffer has `qk_head_dim=576` dimensions:
- **Full 576 dims** are used as **keys** (for Q@K^T score computation)
- **First 512 dims** (kv_lora_rank) are used as **values** (for output computation)

## KV Cache Quantization

| dtype | kv_buffer | kv_scale | Quantization | Bandwidth |
|---|---|---|---|---|
| bf16 | bfloat16 `(total_kv, 1, 576)` | None | No quantization | 1x |
| fp8 | fp8 `(total_kv, 1, 576)` | scalar float32 | Dynamic per-tensor (sglang `scaled_fp8_quant`) | 2x savings |
| mxfp4 | fp4x2 `(total_kv, 1, 288)` | fp8_e8m0 `(total_kv, N_blocks)` | Block-32 MXFP4 (aiter `dynamic_mxfp4_quant`) | 4x savings |

### FP8 quantization (sglang `scaled_fp8_quant`)

- **Granularity**: per-tensor
- **Scale**: `kv_scale = max(abs(kv_bf16)) / fp8_max`
- **Quantize**: `kv_fp8 = (kv_bf16 / kv_scale).clamp(...).to(fp8)`
- **Dequantize**: `kv_bf16 ≈ kv_fp8.to(bf16) * kv_scale`
- **kv_scale**: scalar float32 tensor

### MXFP4 quantization (aiter `dynamic_mxfp4_quant`)

- **Granularity**: per-block of 32 elements
- **FP4 format**: E2M1 — values `[0, 0.5, 1, 1.5, 2, 3, 4, 6]`, max = 6.0
- **Scale format**: E8M0 — exponent-only scale stored in `aiter.dtypes.fp8_e8m0`
- **Packing**: 2 FP4 values packed per byte (low nibble = even index, high nibble = odd index)
- **kv_buffer**: `(total_kv, 1, 288)` in `aiter.dtypes.fp4x2` — packed FP4 data
- **kv_scale**: `(total_kv, N_blocks)` in `aiter.dtypes.fp8_e8m0` — per-block E8M0 scale factors
- **Dequantize**: `aiter.utility.fp4_utils.mxfp4_to_f32` + `e8m0_to_f32` for block-wise scaling

### aiter dtype reference

| Logical type | aiter dtype | PyTorch native (if available) | Fallback |
|---|---|---|---|
| fp4x2 | `aiter.dtypes.fp4x2` | `torch.float4_e2m1fn_x2` | `torch.uint8` |
| fp8_e8m0 | `aiter.dtypes.fp8_e8m0` | `torch.float8_e8m0fnu` | `torch.uint8` |
| fp8 | `aiter.dtypes.fp8` | `torch.float8_e4m3fnuz` (gfx942) / `torch.float8_e4m3fn` (gfx950+) | `torch.uint8` |

## Input

A tuple `(q, kv_data, qo_indptr, kv_indptr, config)`:

```
q:          (total_q, 16, 576)     bfloat16  — absorbed queries
kv_data:    dict with three KV cache formats (see below)
qo_indptr:  (batch_size + 1,)      int32     — query segment pointers
kv_indptr:  (batch_size + 1,)      int32     — KV segment pointers
config:     dict                              — MLA parameters
```

### kv_data dict

All three KV cache formats are provided simultaneously. Each entry is either a
`Tensor` (bf16) or a `(Tensor, Tensor)` tuple (quantized buffer + scale):

```python
kv_data = {
    "bf16":  kv_buffer_bf16,                        # Tensor (total_kv, 1, 576) bfloat16
    "fp8":   (kv_buffer_fp8, kv_scale_fp8),         # (fp8 Tensor, scalar float32)
    "mxfp4": (kv_buffer_mxfp4, kv_scale_mxfp4),    # (fp4x2 Tensor, fp8_e8m0 Tensor)
}
```

### config dict

```python
config = {
    "batch_size": int,
    "num_heads": 16,
    "num_kv_heads": 1,
    "qk_head_dim": 576,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "v_head_dim": 512,
    "q_seq_len": 1,
    "kv_seq_len": int,      # varies per test case (1024 or 8192)
    "sm_scale": 0.04166..., # 1/sqrt(576)
}
```

## Output

```
attention_output: (total_q, 16, 512) bfloat16
```

## Optimization Opportunities

The reference is already a highly optimized aiter a8w8 persistent kernel. To beat it, consider:

1. **MXFP4 KV cache**: 4x bandwidth savings over bf16, 2x over fp8. Two strategies:

   **Strategy A — Fuse dequantization with attention (keep Q in bf16/fp8):**
   Load quantized KV tiles from HBM, dequantize in registers/LDS to bf16, and
   immediately compute QK^T and softmax·V — never writing the decompressed KV back
   to HBM. This eliminates the extra read/write of the bf16 intermediate buffer,
   roughly quartering the memory traffic for mxfp4 compared to the naive
   dequant-then-attend approach.

   **Strategy B — Quantize Q to match KV precision (full low-precision compute):**
   Dynamically quantize Q from bf16 → mxfp4 (per-block scaling), then compute QK^T
   entirely in fp4×fp4 using MFMA instructions on MI355X. The softmax is still done
   in fp32 for numerical stability, and V accumulation uses fp4×fp4 → fp32. This
   trades a small amount of accuracy for significantly higher throughput on the
   matrix units.

2. **Custom split-K / split-batch scheduling**: the aiter kernel uses 32-way KV splits
   with reduce; a different split strategy or tile size may be more efficient for certain
   batch/seq_len combinations.

3. **MQA pattern**: 1 KV head shared across 16 query heads — minimize redundant KV loads
   by loading KV once and broadcasting across all query heads in shared memory/LDS.

4. **Variable-length batching**: indptr-based segmented attention across batch elements.

5. **Split K/V from buffer**: full 576 dims for keys, first 512 for values — potential
   for separate tiling strategies for the score and output stages.

## Accuracy

Submissions are checked against the a8w8 reference with `rtol=2e-02, atol=8e-03`.

Measured accuracy of different approaches vs bf16 torch ground truth:

| Approach | max abs diff | Notes |
|---|---|---|
| aiter a8w8 (reference) | 2.6e-05 — 8.0e-05 | fp8 quantization + kernel accumulation |
| torch fp8 (scaled_mm) | 2e-06 — 1.5e-05 | Closest to bf16 |
| torch mxfp4 | 2.1e-04 — 8.3e-04 | 4-bit quantization noise |

All approaches are well within the tolerance.

## Benchmark Cases

All three KV formats (bf16, fp8, mxfp4) are provided in every test case.

| batch_size | q_seq_len | kv_seq_len |
|---|---|---|
| 4 | 1 | 1024 |
| 4 | 1 | 8192 |
| 32 | 1 | 1024 |
| 32 | 1 | 8192 |
| 64 | 1 | 1024 |
| 64 | 1 | 8192 |
| 256 | 1 | 1024 |
| 256 | 1 | 8192 |

Ranking is by **geometric mean** of benchmark latencies.
