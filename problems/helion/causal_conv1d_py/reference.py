import torch
import torch.nn.functional as F
from task import input_t, output_t
from utils import make_match_reference, DeterministicContext


def generate_input(B: int, D: int, S: int, W: int, seed: int) -> input_t:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    x = torch.randn(B, D, S, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    weight = torch.randn(D, W, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    bias = torch.randn(D, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    return x, weight, bias


def ref_kernel(data: input_t) -> output_t:
    with DeterministicContext():
        x, weight, bias = data
        B, D, S = x.shape
        W = weight.shape[1]

        # Causal (left) padding
        x_padded = F.pad(x, (W - 1, 0))

        # Depthwise conv1d (groups=D)
        output = F.conv1d(
            x_padded,
            weight.unsqueeze(1),  # [D, 1, W]
            bias=bias,
            groups=D,
        )
        return output


check_implementation = make_match_reference(ref_kernel, rtol=1e-2, atol=1e-2)
