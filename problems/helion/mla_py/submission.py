from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    import torch
    import math

    q_nope, q_pe, kv_c, k_pe = data
    B, H, d_c = q_nope.shape
    d_r = q_pe.shape[-1]
    sm_scale = 1.0 / math.sqrt(d_c + d_r)

    score_content = torch.bmm(q_nope.float(), kv_c.float().transpose(-2, -1))
    score_position = torch.bmm(q_pe.float(), k_pe.float().transpose(-2, -1))
    scores = (score_content + score_position) * sm_scale
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.bmm(attn_weights, kv_c.float())
    return output.to(q_nope.dtype)
