from task import input_t, output_t


INT8_MAX = 127
INT8_MIN = -128
INT8_EPS = 1e-10


def custom_kernel(data: input_t) -> output_t:
    import torch

    x, x_q, x_s = data

    x_f32 = x.float()
    absmax = x_f32.abs().amax(dim=-1).clamp(min=INT8_EPS)
    scale = absmax / INT8_MAX
    quantized = torch.round(x_f32 / scale.unsqueeze(-1))
    quantized = quantized.clamp(INT8_MIN, INT8_MAX)

    x_q[...] = quantized
    x_s[...] = scale
    return x_q, x_s
