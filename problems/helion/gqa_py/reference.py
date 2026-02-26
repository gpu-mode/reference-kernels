import torch
from task import input_t, output_t
from utils import make_match_reference, DeterministicContext


def generate_input(B: int, H_q: int, H_kv: int, S: int, D: int, seed: int) -> input_t:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    q = torch.randn(B, H_q, S, D, dtype=torch.float16, device="cuda", generator=gen).contiguous()
    k = torch.randn(B, H_kv, S, D, dtype=torch.float16, device="cuda", generator=gen).contiguous()
    v = torch.randn(B, H_kv, S, D, dtype=torch.float16, device="cuda", generator=gen).contiguous()
    return q, k, v


def ref_kernel(data: input_t) -> output_t:
    with DeterministicContext():
        q, k, v = data
        B, H_q, S, D = q.shape
        H_kv = k.shape[1]
        kv_group_num = H_q // H_kv
        sm_scale = D ** -0.5

        # Expand K and V to match Q's head count
        k_expanded = k.repeat_interleave(kv_group_num, dim=1)
        v_expanded = v.repeat_interleave(kv_group_num, dim=1)

        # Compute attention scores: [B, H_q, S, S]
        attn_scores = torch.matmul(q.float(), k_expanded.float().transpose(-2, -1)) * sm_scale

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=q.device),
            diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        # Softmax and apply to values
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, v_expanded.float())

        return output.to(q.dtype)


check_implementation = make_match_reference(ref_kernel, rtol=1e-2, atol=1e-2)
