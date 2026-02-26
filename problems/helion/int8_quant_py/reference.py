import torch
from task import input_t, output_t
from utils import verbose_allclose

INT8_MAX = 127
INT8_MIN = -128
INT8_EPS = 1e-10


def generate_input(num_tokens: int, hidden_dim: int, seed: int) -> input_t:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    x_q = torch.empty(num_tokens, hidden_dim, dtype=torch.float32, device="cuda").contiguous()
    x_s = torch.empty(num_tokens, dtype=torch.float32, device="cuda").contiguous()
    return x, x_q, x_s


def ref_kernel(data: input_t) -> output_t:
    x, x_q, x_s = data

    x_f32 = x.float()

    # Per-token absmax
    absmax = x_f32.abs().amax(dim=-1).clamp(min=INT8_EPS)

    # Scale = absmax / 127
    scale = absmax / INT8_MAX

    # Quantize
    quantized = torch.round(x_f32 / scale.unsqueeze(-1))
    quantized = quantized.clamp(INT8_MIN, INT8_MAX)

    x_q[...] = quantized
    x_s[...] = scale
    return x_q, x_s


def check_implementation(data, output):
    expected = ref_kernel(data)
    expected_q, expected_s = expected
    received_q, received_s = output

    reasons_s = verbose_allclose(received_s, expected_s, rtol=1e-4, atol=1e-6)
    # Allow +/- 1 LSB for quantized values due to rounding differences
    reasons_q = verbose_allclose(received_q, expected_q, rtol=0, atol=1.0)

    reasons = []
    if reasons_q:
        reasons.append("quantized values mismatch: " + " ".join(reasons_q))
    if reasons_s:
        reasons.append("scales mismatch: " + " ".join(reasons_s))

    if reasons:
        return False, " | ".join(reasons)
    return True, ""
