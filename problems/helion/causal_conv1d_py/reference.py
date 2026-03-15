"""
Causal depthwise Conv1D — reference implementation.

Used in SSM-based LLM architectures such as Mamba, where a short causal
(left-padded) depthwise convolution mixes local context along the sequence
dimension independently per channel, before the selective state-space step.

"Causal" means output position t depends only on input positions <= t,
enforced by padding W-1 zeros on the left and no padding on the right.

This module provides a pure-PyTorch reference against which optimized
Triton/CUDA kernels are verified.

Shapes:
    x      : (B, D, S)  — batch, channels (model dim), sequence length
    weight : (D, W)     — one filter of width W per channel (depthwise)
    bias   : (D,)       — per-channel bias
    output : (B, D, S)  — same shape as input
"""
import torch
import torch.nn.functional as F
from task import input_t, output_t
from utils import make_match_reference, DeterministicContext


def generate_input(B: int, D: int, S: int, W: int, seed: int) -> input_t:
    """Generate random (x, weight, bias) on CUDA with a fixed seed for reproducibility."""
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    x = torch.randn(B, D, S, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    weight = torch.randn(D, W, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    bias = torch.randn(D, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    return x, weight, bias


def ref_kernel(data: input_t) -> output_t:
    """
    Causal depthwise Conv1D via PyTorch.

    Pads W-1 zeros on the left of the sequence so that the convolution at
    position t sees only x[:, :, t-W+1 : t+1], preserving causality.
    groups=D makes each channel use its own filter (depthwise).
    """
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


check_implementation = make_match_reference(ref_kernel, rtol=1e-3, atol=1e-3)
