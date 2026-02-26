from task import input_t, output_t


FP8_MAX = 448.0
FP8_MIN = -448.0
FP8_EPS = 1e-10


def custom_kernel(data: input_t) -> output_t:
    x, x_q, x_s = data
    num_tokens, hidden_dim = x.shape
    num_groups = x_s.shape[1]
    group_size = hidden_dim // num_groups

    x_f32 = x.float()
    x_grouped = x_f32.reshape(num_tokens, num_groups, group_size)

    absmax = x_grouped.abs().amax(dim=-1).clamp(min=FP8_EPS)
    scale = absmax / FP8_MAX
    quantized = (x_grouped / scale.unsqueeze(-1)).clamp(FP8_MIN, FP8_MAX)
    quantized = quantized.reshape(num_tokens, hidden_dim)

    x_q[...] = quantized
    x_s[...] = scale
    return x_q, x_s
