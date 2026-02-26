from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    import torch
    q, k, v = data
    B, H_q, S, D = q.shape
    H_kv = k.shape[1]
    kv_group_num = H_q // H_kv
    sm_scale = D ** -0.5

    k_expanded = k.repeat_interleave(kv_group_num, dim=1)
    v_expanded = v.repeat_interleave(kv_group_num, dim=1)

    attn_scores = torch.matmul(q.float(), k_expanded.float().transpose(-2, -1)) * sm_scale
    causal_mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=q.device), diagonal=1)
    attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
    attn_weights = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_weights, v_expanded.float())
    return output.to(q.dtype)
