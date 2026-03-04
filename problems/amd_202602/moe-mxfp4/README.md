# MXFP4 Mixture-of-Experts (MoE) Fused Kernel

## Description

Implement a DeepSeek-R1 style MXFP4 Mixture-of-Experts (MoE) fused kernel optimized for AMD Instinct MI355X GPU.

The kernel fuses the complete MoE forward pass into a 2-stage pipeline:
1. **Stage 1**: MXFP4 GEMM (gate+up projection) + SwiGLU activation
2. **Stage 2**: MXFP4 GEMM (down projection) + weighted reduction across top-k experts

The reference uses **AITER `fused_moe`** with `QuantType.per_1x32` (MXFP4 block scaling, block_size=32).

## DeepSeek-R1 MoE Architecture

| Parameter | Value | Notes |
|---|---|---|
| hidden_size | 7168 | Model hidden dimension |
| moe_intermediate_size | 2048 | Per-expert intermediate dimension |
| n_routed_experts | 256 | Routed experts (EP-off) or 32 (EP-on, 8-way split) |
| n_shared_experts | 1 | Always selected with weight=1.0 |
| top_k (routed) | 8 | Routed experts per token |
| total_top_k | 9 | 8 routed + 1 shared |
| MoE layers | 58 | Layers 3–60 |

## Kernel Flow

For each token `i` and each assigned expert `j`:

```
(1) Quant activations: hidden_states -> MXFP4 (aiter per-1x32 dynamic quantization)

(2) Stage 1 GEMM + SwiGLU activation:
    gate = x_i @ W_gate_j.T          # [d_hidden] x [d_expert, d_hidden].T -> [d_expert]
    up   = x_i @ W_up_j.T            # [d_hidden] x [d_expert, d_hidden].T -> [d_expert]
    intermediate = SiLU(gate) * up    # SwiGLU activation -> [d_expert]
    (W_gate and W_up are fused as gate_up_weight: one a4w4 GEMM + fused activation)

(3) Stage 2 GEMM:
    expert_out = intermediate @ W_down_j.T  # [d_expert] x [d_hidden, d_expert].T -> [d_hidden]

(4) Weighted reduction:
    output_i += w_ij * expert_out     # accumulate across top_k experts
```

All weight GEMMs are **a4w4** (MXFP4 activations x MXFP4 weights, per-1x32 block scaling).
The AITER CK kernel fuses all of the above into a 2-stage pipeline across all tokens and experts.

## Weight Layout & Pre-shuffling

Weights are provided in two layouts:

| Layout | Description | Use case |
|---|---|---|
| **Raw** | Original MXFP4 quantized weights | PyTorch reference / custom kernels |
| **Pre-shuffled** | `shuffle_weight(w, layout=(16,16))` + `e8m0_shuffle(scale)` | AITER CK kernel (tile-coalesced layout) |

The (16,16) shuffle rearranges weight tiles for coalesced memory access by CK GEMM instructions.
Scale shuffling (`e8m0_shuffle`) reorders E8M0 block scales to match the shuffled weight layout.

You may use either layout — raw weights if you implement your own tiling, or pre-shuffled weights
for direct use with AITER/CK kernels.

## MXFP4 Quantization Details

| Property | Value |
|---|---|
| FP4 format | E2M1 — values `[0, 0.5, 1, 1.5, 2, 3, 4, 6]`, max = 6.0 |
| Scale format | E8M0 — exponent-only (power-of-2 scale) |
| Block size | 32 elements per scale |
| Packing | 2 FP4 values per byte (`fp4x2`): low nibble = even index, high nibble = odd index |
| Padding | Dimensions padded to 256-alignment for CK kernel |

### aiter dtype reference

| Logical type | aiter dtype | PyTorch native (if available) | Fallback |
|---|---|---|---|
| fp4x2 | `aiter.dtypes.fp4x2` | `torch.float4_e2m1fn_x2` | `torch.uint8` |
| fp8_e8m0 | `aiter.dtypes.fp8_e8m0` | `torch.float8_e8m0fnu` | `torch.uint8` |

## Input

A tuple of tensors and a config dict:

```
(hidden_states,
 gate_up_weight, down_weight,                                         # fp4x2 raw
 gate_up_weight_scale, down_weight_scale,                             # e8m0  raw
 gate_up_weight_shuffled, down_weight_shuffled,                       # fp4x2 pre-shuffled
 gate_up_weight_scale_shuffled, down_weight_scale_shuffled,           # e8m0  pre-shuffled
 topk_weights, topk_ids,
 config)
```

### Tensor shapes

| Tensor | Shape | Dtype | Notes |
|---|---|---|---|
| `hidden_states` | `[M, d_hidden]` | bfloat16 | Input activations (M = batch of tokens) |
| `gate_up_weight` | `[E, 2*d_expert_pad, d_hidden_pad//2]` | fp4x2 | Fused gate+up weights, raw |
| `down_weight` | `[E, d_hidden_pad, d_expert_pad//2]` | fp4x2 | Down projection weights, raw |
| `gate_up_weight_scale` | `[E, 2*d_expert_pad, d_hidden_pad//32]` | e8m0 | Block scales for gate_up, raw |
| `down_weight_scale` | `[E, d_hidden_pad, d_expert_pad//32]` | e8m0 | Block scales for down, raw |
| `gate_up_weight_shuffled` | `[E, 2*d_expert_pad, d_hidden_pad//2]` | fp4x2 | Pre-shuffled for CK |
| `down_weight_shuffled` | `[E, d_hidden_pad, d_expert_pad//2]` | fp4x2 | Pre-shuffled for CK |
| `gate_up_weight_scale_shuffled` | `[padded, flat]` | e8m0 | Pre-shuffled for CK |
| `down_weight_scale_shuffled` | `[padded, flat]` | e8m0 | Pre-shuffled for CK |
| `topk_weights` | `[M, total_top_k]` | float32 | Routing weights |
| `topk_ids` | `[M, total_top_k]` | int32 | Expert indices (see below) |

### topk_ids format

- First `n_experts_per_token` columns: routed expert IDs `[0, n_routed_experts)`
- Last `n_shared_experts` columns: shared expert IDs `[n_routed_experts, n_routed_experts + n_shared_experts)`
- Shared experts are always selected with weight = 1.0

### config dict

```python
config = {
    "d_hidden": int,          # hidden dimension (e.g. 7168)
    "d_expert": int,          # expert intermediate dimension (e.g. 2048 or 256)
    "d_hidden_pad": int,      # d_hidden padded to 256-alignment
    "d_expert_pad": int,      # d_expert padded to 256-alignment
    "n_routed_experts": int,  # number of routed experts
    "n_shared_experts": int,  # number of shared experts (1)
    "n_experts_per_token": int, # routed top-k (8)
    "total_top_k": int,       # routed + shared (9)
    "bs": int,                # batch size (number of tokens)
}
```

## Output

```
output: [M, d_hidden] bfloat16
```

## Reference Performance

AITER `fused_moe` with MXFP4 (E includes shared expert, top_k = routed + shared):

| bs | E | d_hidden | d_expert | top_k | time (us) |
|---|---|---|---|---|---|
| 4 | 257 | 7168 | 256 | 9 | 46.9 |
| 64 | 257 | 7168 | 256 | 9 | 187.7 |
| 256 | 257 | 7168 | 256 | 9 | 245.7 |
| 64 | 33 | 7168 | 2048 | 9 | 220.6 |
| 256 | 33 | 7168 | 2048 | 9 | 276.4 |
| 1024 | 33 | 7168 | 2048 | 9 | 572.2 |

## Optimization Opportunities

The AITER CK `fused_moe` kernel is already well-optimized. To beat it, consider:

1. **Custom tiling / scheduling**: The CK kernel uses a fixed tile strategy. For small batch sizes
   (bs=4) or highly skewed expert distributions, a custom schedule may reduce idle waves.

2. **Activation quantization fusion**: The reference quantizes activations separately before the
   GEMM. Fusing dynamic MXFP4 quantization into the Stage 1 GEMM prologue saves one global
   memory round-trip.

3. **Inter-stage fusion**: The reference runs Stage 1 and Stage 2 as separate kernel launches.
   Fusing both stages (gate_up GEMM → SwiGLU → down GEMM → accumulate) into a single kernel
   eliminates the intermediate buffer write/read between stages.

4. **Expert-parallel wave scheduling**: With 257 experts but only 9 active per token, most
   expert slots are empty. A work-stealing or compact-dispatch strategy can minimize wasted
   wavefronts.

5. **Shared expert fusion**: The shared expert is always selected for all tokens. It could be
   computed as a dense GEMM (no routing overhead) and fused with the routed expert reduction.

6. **Split-K for large M**: For bs=1024 with EP-on (E=33, d_expert=2048), the GEMMs are large
   enough to benefit from split-K parallelism within each expert.

## Accuracy

Submissions are checked against the AITER reference with `rtol=1e-2, atol=1e-2`.

## Benchmark Cases

### EP-off (all 257 experts on 1 GPU, d_expert=256)

| bs | E | d_hidden | d_expert | top_k |
|---|---|---|---|---|
| 4 | 257 | 7168 | 256 | 9 |
| 64 | 257 | 7168 | 256 | 9 |
| 256 | 257 | 7168 | 256 | 9 |

### EP-on (EP=8, 33 experts per GPU, d_expert=2048)

| bs | E | d_hidden | d_expert | top_k |
|---|---|---|---|---|
| 64 | 33 | 7168 | 2048 | 9 |
| 256 | 33 | 7168 | 2048 | 9 |
| 1024 | 33 | 7168 | 2048 | 9 |

Ranking is by **geometric mean** of benchmark latencies.
