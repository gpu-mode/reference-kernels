from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    import torch

    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = A.shape[-1]

    w = torch.empty_like(k)
    u = torch.empty_like(v)

    for cs in range(0, T, BT):
        ce = cs + BT
        A_bh = A[:, cs:ce, :, :].permute(0, 2, 1, 3).float()

        vb = (v[:, cs:ce, :, :] * beta[:, cs:ce, :, None]).permute(0, 2, 1, 3).float()
        u[:, cs:ce, :, :] = torch.matmul(A_bh, vb).permute(0, 2, 1, 3)

        kb = (k[:, cs:ce, :, :] * beta[:, cs:ce, :, None] * torch.exp(g[:, cs:ce, :, None])).permute(0, 2, 1, 3).float()
        w[:, cs:ce, :, :] = torch.matmul(A_bh, kb).permute(0, 2, 1, 3)

    return w, u
