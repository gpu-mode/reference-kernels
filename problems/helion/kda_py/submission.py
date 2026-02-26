from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    import torch

    q, k, v, g, beta = data
    B, T, H, K = q.shape
    V_dim = v.shape[-1]
    scale = K ** -0.5

    q_r = q.permute(0, 2, 1, 3).reshape(B * H, T, K)
    k_r = k.permute(0, 2, 1, 3).reshape(B * H, T, K)
    v_r = v.permute(0, 2, 1, 3).reshape(B * H, T, V_dim)
    g_r = g.permute(0, 2, 1, 3).reshape(B * H, T, K)
    beta_r = beta.permute(0, 2, 1).reshape(B * H, T)

    S = torch.zeros(B * H, K, V_dim, dtype=torch.float32, device=q.device)
    outputs = []

    for t in range(T):
        decay = torch.exp(g_r[:, t, :])
        S = S * decay.unsqueeze(-1)
        k_t = k_r[:, t, :]
        predicted = torch.bmm(k_t.unsqueeze(1), S).squeeze(1)
        v_t = v_r[:, t, :]
        delta = v_t - predicted
        b_t = beta_r[:, t].unsqueeze(-1)
        correction = torch.bmm(k_t.unsqueeze(-1), (b_t * delta).unsqueeze(1))
        S = S + correction
        q_t = q_r[:, t, :]
        o_t = scale * torch.bmm(q_t.unsqueeze(1), S).squeeze(1)
        outputs.append(o_t)

    output = torch.stack(outputs, dim=1)
    output = output.reshape(B, H, T, V_dim).permute(0, 2, 1, 3).contiguous()
    return output
