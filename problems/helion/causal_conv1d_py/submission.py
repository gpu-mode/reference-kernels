from task import input_t, output_t

import torch
import helion
import helion.language as hl


# NOTE: This is an intentionally inefficient baseline implementation.
@helion.kernel(
    static_shapes=True,
    config=helion.Config(block_sizes=[1, 8], num_warps=1, num_stages=1),
)
def conv1d_kernel(
    x_pad: torch.Tensor,  # (B, D, L) zero-padded input
    w: torch.Tensor,      # (D, W) filter coefficients
    b: torch.Tensor,      # (D,) additive offset
) -> torch.Tensor:
    B = x_pad.size(0)
    D = x_pad.size(1)
    L = x_pad.size(2)
    W = hl.specialize(w.size(1))
    N = L - W + 1

    y = torch.empty(B, D, N, dtype=x_pad.dtype, device=x_pad.device)

    for rb, rd, rs in hl.tile([B, D, N], block_size=[1, None, None]):
        bi = rb.begin
        acc1 = hl.zeros([rd, rs], dtype=torch.float32)
        acc2 = hl.zeros([rd, rs], dtype=torch.float32)
        acc3 = hl.zeros([rd, rs], dtype=torch.float32)
        for j in range(W):
            c1 = w[rd, j].to(torch.float32)
            x1 = hl.load(x_pad, [bi, rd, rs.index + j]).to(torch.float32)
            acc1 = acc1 + x1 * c1[:, None]
            c2 = w[rd, j].to(torch.float32)
            x2 = hl.load(x_pad, [bi, rd, rs.index + j]).to(torch.float32)
            acc2 = acc2 + x2 * c2[:, None]
            c3 = w[rd, j].to(torch.float32)
            x3 = hl.load(x_pad, [bi, rd, rs.index + j]).to(torch.float32)
            acc3 = acc3 + x3 * c3[:, None]
        acc = (acc1 + acc2 + acc3) / 3.0
        acc = acc + b[rd].to(torch.float32)[:, None]
        y[rb, rd, rs] = acc[None, :, :].to(y.dtype)

    return y


def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data
    W = weight.shape[1]
    pad_zeros = torch.zeros(x.shape[0], x.shape[1], W - 1, dtype=x.dtype, device=x.device)
    padded = torch.cat([pad_zeros, x], dim=2)
    return conv1d_kernel(padded, weight, bias)
