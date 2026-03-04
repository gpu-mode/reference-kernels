from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    import torch

    q, k, v_new, h, g = data
    B, T, H, K = q.shape
    V = v_new.shape[-1]
    BT = 64
    scale = K ** -0.5

    o = torch.empty_like(v_new)
    causal = torch.tril(torch.ones(BT, BT, device=q.device, dtype=torch.bool))

    for cs in range(0, T, BT):
        ce = cs + BT
        c_idx = cs // BT

        b_q = q[:, cs:ce, :, :].permute(0, 2, 1, 3).float()
        b_k = k[:, cs:ce, :, :].permute(0, 2, 1, 3).float()
        b_v = v_new[:, cs:ce, :, :].permute(0, 2, 1, 3).float()
        b_h = h[:, c_idx, :, :, :].float()
        b_g = g[:, cs:ce, :].permute(0, 2, 1).float()

        inter = torch.matmul(b_q, b_h)
        inter = inter * torch.exp(b_g).unsqueeze(-1)

        attn = torch.matmul(b_q, b_k.transpose(-1, -2))
        g_diff = b_g.unsqueeze(-1) - b_g.unsqueeze(-2)
        attn = attn * torch.exp(g_diff)
        attn = attn.masked_fill(~causal, 0.0)
        intra = torch.matmul(attn, b_v)

        b_o = (inter + intra) * scale
        o[:, cs:ce, :, :] = b_o.permute(0, 2, 1, 3)

    return o
