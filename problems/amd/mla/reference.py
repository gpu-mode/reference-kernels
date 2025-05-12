import torch
import torch.nn as nn
import math
import random
from task import input_t, output_t
from utils import make_match_reference
from typing import Optional, Tuple, Union

def generate_input(b, d, dv, hq, sq, hkv, meansk, seed):

    cache_seqlens = torch.full((b,), meansk, dtype=torch.int32)
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = math.ceil(max_seqlen/256) * 256

    gen = torch.Generator()
    gen.manual_seed(seed)

    q = torch.randn((b, sq, hq, d), dtype=torch.bfloat16, generator=gen)
    k = torch.randn((b, max_seqlen_pad, hkv, d), dtype=torch.bfloat16, generator=gen)
    v = torch.randn((b, max_seqlen_pad, hkv, dv), dtype=torch.bfloat16, generator=gen)
    return q, k, v, cache_seqlens, max_seqlen_pad

def scaled_dot_product_attention(query, key, value, hq, hkv, is_causal=False):
    key = key.repeat_interleave(hq // hkv, dim=0)
    value = value.repeat_interleave(hq // hkv, dim=0)

    scale = 1.0 / math.sqrt(query.size(-1))
    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Apply causal mask if needed
    if is_causal:
        sq = query.shape[-2]
        sk = key.shape[-2]
        attn_bias = torch.zeros(sq, sk, dtype=torch.float32, device=query.device)
        temp_mask = torch.ones(sq, sk, dtype=torch.bool, device=query.device).tril(diagonal=sk - sq)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_weight += attn_bias

    attn_weight = torch.nn.functional.softmax(attn_weight, dim=-1)
    out = torch.matmul(attn_weight, value)

    return out

def ref_kernel(data: input_t) -> output_t:
    """
    q shape: batch_size, q_seqlen, hq, d
    k shape: batch_size, max_seqlen_pad, hkv, d
    v shape: batch_size, max_seqlen_pad, hkv, d_v
    cache_seqlens: tensor containing the actual sequence lengths
    max_seqlen_pad: the padded sequence length
    positions: tensor containing position information for RoPE
    """
    q, k, v, cache_seqlens, max_seqlen_pad = data
    causal = False
    b, sq, hq, d = q.shape
    _, _, hkv, dv = v.shape
    out = torch.empty(b, sq, hq, dv, dtype=q.dtype)
    for i in range(b):
        begin = i * max_seqlen_pad
        end = begin + cache_seqlens[i]
        ik = k.view(-1, hkv, d)[begin:end]
        iv = v.view(-1, hkv, dv)[begin:end]
        iq = q[i]
        O = scaled_dot_product_attention(
                iq.transpose(0, 1),
                ik.transpose(0, 1),
                iv.transpose(0, 1),
                hq=hq,
                hkv=hkv,
                is_causal=causal,
            )
        out[i] = O.transpose(0, 1)
    return out

check_implementation = make_match_reference(ref_kernel)