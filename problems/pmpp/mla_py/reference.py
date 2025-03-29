import torch
from task import input_t, output_t
from torch.nn.functional import softmax
from utils import make_match_reference

def generate_input(B, S, H_Q, H_KV, D, D_V) -> input_t:
    dtype = torch.bfloat16
    seq_len = S  # This represents the number of tokens already in the sequence
    total_tokens = B * seq_len

    # q represents the new token being generated, one per batch
    q = torch.randn(B, H_Q, D, dtype=dtype, device="cuda")

    # k_buffer and v_buffer represent all previous tokens
    k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")
    v_buffer = torch.randn(total_tokens, H_KV, D_V, dtype=dtype, device="cuda")
    return (q, k_buffer, v_buffer)

def ref_kernel(data: input_t) -> output_t:
    q, k_buffer, v_buffer = data
    B, H_Q, D = q.shape
    total_tokens, H_KV, D_V = v_buffer.shape
    seq_len = total_tokens // B
    assert total_tokens % B == 0
    sm_scale = 1.0 / (D**0.5)
    # k_buffer, v_buffer to split
    k = k_buffer.reshape(B, seq_len, H_KV, D)
    v = v_buffer.reshape(B, seq_len, H_KV, D_V)
    q_torch = q.reshape(B, 1, H_Q, D).permute(0,2,1,3)
    #k = k.transpose(0,1,3,2)
    enable_GQA = (H_Q != H_KV)
    if enable_GQA:
        num_repeat = H_Q // H_KV
        k = k.repeat_interleave(num_repeat, dim=2)
        v = v.repeat_interleave(num_repeat, dim=2)
    k = k.permute(0,2,3,1)
    v = v.permute(0,2,1,3)
    attn_s = sm_scale * (q_torch@k)
    attn_s = softmax(attn_s, dim=-1)
    native_o = attn_s @ v
    return native_o

check_implementation = make_match_reference(ref_kernel)

