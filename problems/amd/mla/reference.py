import torch
import math
import random
from task import input_t, output_t
from rotary_embedding import DeepseekScalingRotaryEmbedding
from utils import make_match_reference

def generate_input(b, s_q, mean_sk, h_q, h_kv, d, dv, causal, varlen, seed):
    print(
        f"{b=}, {s_q=}, {mean_sk=}, {h_q=}, {h_kv=}, {d=}, {dv=}, {causal=}, {varlen=}"
    )

    cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32)
    if varlen:
        for i in range(b):
            cache_seqlens[i] = max(random.normalvariate(mean_sk, mean_sk / 2), s_q)
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = math.ceil(max_seqlen/256) * 256

    gen = torch.Generator()
    gen.manual_seed(seed)
    
    q = torch.randn((b, s_q, h_q, d), dtype=torch.bfloat16, generator=gen)
    k = torch.randn((b, max_seqlen_pad, h_kv, d), dtype=torch.bfloat16, generator=gen)
    v = torch.randn((b, max_seqlen_pad, h_kv, dv), dtype=torch.bfloat16, generator=gen)
    positions = (
        torch.tensor([s_q], device=q.device).unsqueeze(0).repeat(b, 1)
    ) # only gen 1 token per req
    return q, k, v, cache_seqlens, max_seqlen_pad, positions, causal

def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
    query = query.float()
    key = key.float()
    value = value.float()
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight @ value

def ref_kernel(data: input_t, use_rope=False) -> output_t:
    """
    q shape: batch_size, q_seqlen, h_q, d
    k shape: batch_size, max_seqlen_pad, h_kv, d
    v shape: batch_size, max_seqlen_pad, h_kv, d_v
    """
    q, k, v, cache_seqlens, max_seqlen_pad, positions, causal = data
    b, s_q, h_q, d = q.shape
    _, _, h_kv, dv = v.shape
    rope_head_dim = d - dv 
    rotary_dim = rope_head_dim 
    rope_max_seq_len=16324
    rope_base=1000
    rope_scaling=1.0
    is_neox_style=True
    rotary_emb = DeepseekScalingRotaryEmbedding(
                    rope_head_dim, 
                    rotary_dim, 
                    rope_max_seq_len, 
                    rope_base, 
                    is_neox_style,
                    rope_scaling,
                    q.dtype, 
                    device=q.device)
    out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32)
    for i in range(b):
        begin = i * max_seqlen_pad
        end = begin + cache_seqlens[i]
        ik = k.view(-1, h_kv, d)[begin:end]
        iv = v.view(-1, h_kv, dv)[begin:end]
        iq = q[i]
        if use_rope:
            q_nope, q_pe = iq.split([dv, rotary_dim], dim=-1) # [s_q, h_q, d]
            k_nope, k_pe = ik.split([dv, rotary_dim], dim=-1) # [s_k, h_kv, d]
            q_pe, k_pe = rotary_emb(positions[i], q_pe, k_pe)
            iq[..., dv:]=q_pe
            ik[..., dv:]=k_pe       
        O = scaled_dot_product_attention(
                iq.transpose(0, 1),
                ik.transpose(0, 1),
                iv.transpose(0, 1),
                h_q=h_q,
                h_kv=h_kv,
                is_causal=causal,
            )
        out[i] = O.transpose(0, 1)
    return out


check_implementation = make_match_reference(ref_kernel)