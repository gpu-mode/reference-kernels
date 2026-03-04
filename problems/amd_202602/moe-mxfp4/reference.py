from utils import make_match_reference
from task import input_t, output_t
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
from aiter.utility import fp4_utils
from aiter.ops.shuffle import shuffle_weight


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
MXFP4_BLOCK_SIZE = 32
PAD_ALIGN = 256


def _pad_to(x: int, align: int) -> int:
    return (x + align - 1) // align * align


# ──────────────────────────────────────────────────────────────────────
# generate_input: produce all tensors needed by ref_kernel
#
# Models DeepSeek-R1 MoE layer shapes:
#   - d_hidden = 7168
#   - d_expert = moe_intermediate_size (full=2048, or TP-split)
#   - E_total = n_routed_experts + n_shared_experts (257 or 33)
#   - top_k_total = nexpertspertoken + nsharedexperts (8+1=9)
#
# ──────────────────────────────────────────────────────────────────────
def generate_input(
    dhidden: int,
    dexpert: int,
    nroutedexperts: int,
    nexpertspertoken: int,
    nsharedexperts: int,
    bs: int,
    seed: int,
) -> input_t:
    d_hidden = dhidden
    d_expert = dexpert
    n_routed_experts = nroutedexperts
    n_shared_experts = nsharedexperts
    routed_top_k = nexpertspertoken
    total_top_k = routed_top_k + n_shared_experts   # e.g. 8 + 1 = 9
    E_total = n_routed_experts + n_shared_experts    # e.g. 256 + 1 = 257
    M = bs  # number of tokens

    # Padded dimensions (AITER MXFP4 requires 256-alignment)
    d_hidden_pad = _pad_to(d_hidden, PAD_ALIGN)
    d_expert_pad = _pad_to(d_expert, PAD_ALIGN)

    config = {
        "d_hidden": d_hidden,
        "d_expert": d_expert,
        "d_hidden_pad": d_hidden_pad,
        "d_expert_pad": d_expert_pad,
        "n_routed_experts": n_routed_experts,
        "n_shared_experts": n_shared_experts,
        "n_experts_per_token": routed_top_k,
        "total_top_k": total_top_k,
        "bs": M,
    }

    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)

    # ── hidden_states [M, d_hidden] ──
    hidden_states = torch.randn(
        (M, d_hidden), device='cuda', dtype=torch.bfloat16, generator=gen,
    )

    # ── Router: softmax top-k (routed experts only) ──
    router_weight = torch.randn(
        (n_routed_experts, d_hidden), device='cuda', dtype=torch.bfloat16, generator=gen,
    ) / math.sqrt(d_hidden)
    router_logits = F.linear(hidden_states, router_weight)  # [M, n_routed_experts]
    scores = router_logits.softmax(dim=-1)
    routed_weights, routed_ids = torch.topk(
        scores, k=routed_top_k, dim=-1, sorted=False
    )
    routed_weights = routed_weights.to(torch.float32)
    routed_ids = routed_ids.to(torch.int32)

    # ── Append shared expert(s): always selected, weight = 1.0 ──
    # Shared experts are indexed as n_routed_experts, n_routed_experts+1, ...
    shared_ids = torch.arange(
        n_routed_experts, E_total, device='cuda', dtype=torch.int32
    ).unsqueeze(0).expand(M, -1)                               # [M, n_shared_experts]
    shared_weights = torch.ones(
        (M, n_shared_experts), device='cuda', dtype=torch.float32
    )

    topk_ids = torch.cat([routed_ids, shared_ids], dim=-1)        # [M, total_top_k]
    topk_weights = torch.cat([routed_weights, shared_weights], dim=-1)  # [M, total_top_k]

    # ── Expert weights: bf16 -> quantize to MXFP4 ──
    # Generate weights for ALL experts (routed + shared)
    # gate_up = fused [gate_proj; up_proj] per expert: [2*d_expert_pad, d_hidden_pad]
    # down    = down_proj                  per expert: [d_hidden_pad, d_expert_pad]
    gate_up_q_list, gate_up_s_list = [], []
    down_q_list, down_s_list = [], []

    for _ in range(E_total):
        # gate_proj + up_proj -> fused [2*d_expert_pad, d_hidden_pad]
        gate_bf16 = torch.randn(
            (d_expert_pad, d_hidden_pad), device='cuda', dtype=torch.bfloat16, generator=gen
        ) / math.sqrt(d_hidden)
        up_bf16 = torch.randn(
            (d_expert_pad, d_hidden_pad), device='cuda', dtype=torch.bfloat16, generator=gen
        ) / math.sqrt(d_hidden)
        gate_up_bf16 = torch.cat([gate_bf16, up_bf16], dim=0)

        # down_proj -> [d_hidden_pad, d_expert_pad]
        down_bf16 = torch.randn(
            (d_hidden_pad, d_expert_pad), device='cuda', dtype=torch.bfloat16, generator=gen
        ) / math.sqrt(d_expert)

        # Quantize to MXFP4
        gu_q, gu_s = fp4_utils.dynamic_mxfp4_quant(gate_up_bf16)
        dn_q, dn_s = fp4_utils.dynamic_mxfp4_quant(down_bf16)

        gate_up_q_list.append(gu_q)
        gate_up_s_list.append(gu_s)
        down_q_list.append(dn_q)
        down_s_list.append(dn_s)

    # Stack into [E_total, ...] tensors — raw (before shuffle)
    gate_up_weight = torch.stack(gate_up_q_list)          # [E_total, 2*d_expert_pad, d_hidden_pad//2]  fp4x2
    gate_up_weight_scale = torch.stack(gate_up_s_list)    # [E_total, 2*d_expert_pad, scale_K]          e8m0
    down_weight = torch.stack(down_q_list)                # [E_total, d_hidden_pad, d_expert_pad//2]    fp4x2
    down_weight_scale = torch.stack(down_s_list)          # [E_total, d_hidden_pad, scale_K]            e8m0

    # Pre-shuffled weight. You can also shuffle the weights yourself before calling the kernel.
    gate_up_weight_shuffled = shuffle_weight(gate_up_weight.clone())
    down_weight_shuffled = shuffle_weight(down_weight.clone())
    gate_up_weight_scale_shuffled = fp4_utils.e8m0_shuffle(gate_up_weight_scale.reshape(E_total, -1))
    down_weight_scale_shuffled = fp4_utils.e8m0_shuffle(down_weight_scale.reshape(E_total, -1))

    return (
        hidden_states,                  # [M, d_hidden]                              bf16
        gate_up_weight,                 # [E_total, 2*d_expert_pad, d_hidden_pad//2] fp4x2  (raw)
        down_weight,                    # [E_total, d_hidden_pad, d_expert_pad//2]   fp4x2  (raw)
        gate_up_weight_scale,           # [E_total, 2*d_expert_pad, scale_K]         e8m0   (raw)
        down_weight_scale,              # [E_total, d_hidden_pad, scale_K]           e8m0   (raw)
        gate_up_weight_shuffled,        # [E_total, 2*d_expert_pad, d_hidden_pad//2] fp4x2  (pre-shuffled)
        down_weight_shuffled,           # [E_total, d_hidden_pad, d_expert_pad//2]   fp4x2  (pre-shuffled)
        gate_up_weight_scale_shuffled,  # [padded, flat]                             e8m0   (pre-shuffled)
        down_weight_scale_shuffled,     # [padded, flat]                             e8m0   (pre-shuffled)
        topk_weights,                   # [M, total_top_k]                           float32
        topk_ids,                       # [M, total_top_k]                           int32
        config,
    )




# ──────────────────────────────────────────────────────────────────────
# ref_kernel_pytorch: pure PyTorch implementation (dequant + matmul)
# ──────────────────────────────────────────────────────────────────────
def _dequant_mxfp4(weight_fp4, scale_e8m0):
    """
    Dequantize MXFP4 weight to float32.

    weight_fp4:  [N, K//2]  fp4x2  (raw, not shuffled)
    scale_e8m0:  [padded_N, ceil(K/32)] e8m0   (M-dim padded to 256-align by dynamic_mxfp4_quant)

    Returns: [N, K] float32
    """
    # fp4x2 -> float32 lookup: [N, K]
    w_f32 = fp4_utils.mxfp4_to_f32(weight_fp4)            # [N, K]
    # e8m0 -> float32 power-of-2 scale: [padded_N, scale_K]
    s_f32 = fp4_utils.e8m0_to_f32(scale_e8m0)             # [padded_N, scale_K]
    N, K = w_f32.shape
    # Trim scale rows to match weight rows (scale M-dim is padded to 256)
    s_f32 = s_f32[:N, :]
    # Broadcast scale across block_size=32 columns
    s_f32 = s_f32.repeat_interleave(MXFP4_BLOCK_SIZE, dim=-1)[:, :K]  # [N, K]
    return w_f32 * s_f32

# ──────────────────────────────────────────────────────────────────────
# ref_kernel_pytorch: pure PyTorch implementation (dequant + matmul)
# will not run. only for reference
# ──────────────────────────────────────────────────────────────────────
def ref_kernel_pytorch(data: input_t) -> output_t:
    """
    Pure PyTorch reference: dequantize MXFP4 weights -> bf16 matmul -> SwiGLU -> matmul.
    Uses the raw (un-shuffled) weights.
    """
    (
        hidden_states,             # [M, d_hidden]          bf16
        gate_up_weight,            # [E, 2*d_expert_pad, d_hidden_pad//2]  fp4x2
        down_weight,               # [E, d_hidden_pad, d_expert_pad//2]    fp4x2
        gate_up_weight_scale,      # [E, 2*d_expert_pad, scale_K]          e8m0
        down_weight_scale,         # [E, d_hidden_pad, scale_K]            e8m0
        gate_up_weight_shuffled,
        down_weight_shuffled,
        gate_up_weight_scale_shuffled,
        down_weight_scale_shuffled,
        topk_weights,              # [M, top_k]  float32
        topk_ids,                  # [M, top_k]  int32
        config,
    ) = data

    d_hidden = config["d_hidden"]
    d_expert = config["d_expert"]
    d_hidden_pad = config["d_hidden_pad"]
    d_expert_pad = config["d_expert_pad"]
    M = hidden_states.shape[0]
    top_k = topk_ids.shape[1]
    E = gate_up_weight.shape[0]

    # Dequantize all expert weights to float32
    # gate_up: [E, 2*d_expert_pad, d_hidden_pad] -> trim to [E, 2*d_expert, d_hidden]
    # down:    [E, d_hidden_pad, d_expert_pad]    -> trim to [E, d_hidden, d_expert]
    gate_up_dq = torch.stack([
        _dequant_mxfp4(gate_up_weight[e], gate_up_weight_scale[e])
        for e in range(E)
    ])  # [E, 2*d_expert_pad, d_hidden_pad]
    gate_up_dq = gate_up_dq[:, :2 * d_expert, :d_hidden].to(torch.bfloat16)

    down_dq = torch.stack([
        _dequant_mxfp4(down_weight[e], down_weight_scale[e])
        for e in range(E)
    ])  # [E, d_hidden_pad, d_expert_pad]
    down_dq = down_dq[:, :d_hidden, :d_expert].to(torch.bfloat16)

    # Split gate_up -> gate [E, d_expert, d_hidden], up [E, d_expert, d_hidden]
    gate_w, up_w = gate_up_dq.chunk(2, dim=1)  # each [E, d_expert, d_hidden]

    # Per-token MoE forward
    output = torch.zeros((M, d_hidden), dtype=torch.bfloat16, device=hidden_states.device)

    for i in range(M):
        x = hidden_states[i]  # [d_hidden]
        for k in range(top_k):
            eid = topk_ids[i, k].item()
            w = topk_weights[i, k].item()

            # Stage 1: gate_proj + up_proj + SwiGLU
            gate_out = F.silu(x @ gate_w[eid].T)     # [d_expert]
            up_out = x @ up_w[eid].T                  # [d_expert]
            intermediate = gate_out * up_out           # [d_expert]

            # Stage 2: down_proj
            # down_dq[eid] is [d_hidden, d_expert], .T is [d_expert, d_hidden]
            expert_out = intermediate @ down_dq[eid].T  # [d_hidden]

            output[i] += w * expert_out

    return output



# ──────────────────────────────────────────────────────────────────────
# ref_kernel: calls AITER fused_moe with MXFP4 quantized weights
# ──────────────────────────────────────────────────────────────────────
def ref_kernel(data: input_t) -> output_t:
    """
    Reference implementation using AITER's fused_moe kernel with MXFP4 quantized weights.

    Input data tuple (E = n_routed_experts + n_shared_experts, total_top_k = routed + shared):
        hidden_states:                [M, d_hidden]                           bf16
        gate_up_weight:               [E, 2*d_expert_pad, d_hidden_pad//2]    fp4x2  (raw, before shuffle)
        down_weight:                  [E, d_hidden_pad, d_expert_pad//2]      fp4x2  (raw, before shuffle)
        gate_up_weight_scale:         [E, 2*d_expert_pad, scale_K]            e8m0   (raw, before shuffle)
        down_weight_scale:            [E, d_hidden_pad, scale_K]              e8m0   (raw, before shuffle)
        gate_up_weight_shuffled:      [E, 2*d_expert_pad, d_hidden_pad//2]    fp4x2  (pre-shuffled)
        down_weight_shuffled:         [E, d_hidden_pad, d_expert_pad//2]      fp4x2  (pre-shuffled)
        gate_up_weight_scale_shuffled:[padded, flat]                          e8m0   (pre-shuffled)
        down_weight_scale_shuffled:   [padded, flat]                          e8m0   (pre-shuffled)
        topk_weights:                 [M, total_top_k]                        float32
        topk_ids:                     [M, total_top_k]                        int32
        config:                       dict

    Returns:
        output: [M, d_hidden] bf16
    """
    (
        hidden_states,
        gate_up_weight,
        down_weight,
        gate_up_weight_scale,
        down_weight_scale,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        gate_up_weight_scale_shuffled,
        down_weight_scale_shuffled,
        topk_weights,
        topk_ids,
        config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    output = fused_moe(
        hidden_states,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        topk_weights,
        topk_ids,
        expert_mask=None,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,  # MXFP4 uses per_1x32 block scaling
        doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None,
        a2_scale=None,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
    )

    return output



check_implementation = make_match_reference(ref_kernel, rtol=5e-2, atol=5e-2)
