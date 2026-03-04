import torch
from task import input_t, output_t
from utils import verbose_allclose

FP8_MAX = 448.0
FP8_MIN = -448.0
FP8_EPS = 1e-10


def generate_input(num_tokens: int, hidden_dim: int, group_size: int, seed: int) -> input_t:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    x_q = torch.empty(num_tokens, hidden_dim, dtype=torch.float32, device="cuda").contiguous()
    x_s = torch.empty(num_tokens, hidden_dim // group_size, dtype=torch.float32, device="cuda").contiguous()
    return x, x_q, x_s


def ref_kernel(data: input_t) -> output_t:
    x, x_q, x_s = data
    num_tokens, hidden_dim = x.shape
    num_groups = x_s.shape[1]
    group_size = hidden_dim // num_groups

    x_f32 = x.float()
    x_grouped = x_f32.reshape(num_tokens, num_groups, group_size)

    # Per-group absmax
    absmax = x_grouped.abs().amax(dim=-1).clamp(min=FP8_EPS)

    # Scale = absmax / fp8_max
    scale = absmax / FP8_MAX

    # Quantize
    quantized = (x_grouped / scale.unsqueeze(-1)).clamp(FP8_MIN, FP8_MAX)
    quantized = quantized.reshape(num_tokens, hidden_dim)

    x_q[...] = quantized
    x_s[...] = scale
    return x_q, x_s


def check_implementation(data, output):
    expected = ref_kernel(data)
    expected_q, expected_s = expected
    received_q, received_s = output

    reasons_q = verbose_allclose(received_q, expected_q, rtol=1e-3, atol=1e-3)
    reasons_s = verbose_allclose(received_s, expected_s, rtol=1e-4, atol=1e-6)

    reasons = []
    if reasons_q:
        reasons.append("quantized values mismatch: " + " ".join(reasons_q))
    if reasons_s:
        reasons.append("scales mismatch: " + " ".join(reasons_s))

    if reasons:
        return False, " | ".join(reasons)
    return True, ""
