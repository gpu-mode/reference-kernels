from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    import torch

    k, w, u, g = data
    B, T, H, K = k.shape
    V = u.shape[-1]
    BT = 64
    NT = T // BT

    h = torch.empty(B, NT, H, K, V, dtype=torch.float32, device=k.device)
    v_new = torch.empty_like(u)

    for b in range(B):
        for hh in range(H):
            b_h = torch.zeros(K, V, dtype=torch.float32, device=k.device)

            for c in range(NT):
                cs = c * BT
                ce = cs + BT

                h[b, c, hh] = b_h

                b_w = w[b, cs:ce, hh].float()
                b_u = u[b, cs:ce, hh].float()
                b_v = b_u - torch.matmul(b_w, b_h)
                v_new[b, cs:ce, hh] = b_v

                b_g = g[b, cs:ce, hh].float()
                b_g_last = b_g[-1]
                b_v_gated = b_v * torch.exp(b_g_last - b_g)[:, None]

                b_h = b_h * torch.exp(b_g_last)
                b_k = k[b, cs:ce, hh].float()
                b_h = b_h + torch.matmul(b_k.T, b_v_gated)

    return h, v_new
