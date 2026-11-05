import torch
from task import input_t, output_t
from utils import DeterministicContext, match_reference


def ref_kernel(data: input_t) -> output_t:
    """
    Reference implementation of inclusive prefix sum using PyTorch.
    Args:
        data: Input tensor to compute prefix sum on
    Returns:
        Tensor containing the inclusive prefix sum
    """
    with DeterministicContext():
        data, output = data
        output = torch.cumsum(data.to(torch.float64), dim=0).to(torch.float32)
        return output


def generate_input(
    size: int, seed: int, distribution: str = "normal", *, device="cuda"
) -> input_t:
    """
    Generates random input tensor.
    Returns:
        Tensor to compute prefix sum on
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    x = torch.randn(
        size, device=device, dtype=torch.float32, generator=gen
    ).contiguous()
    if distribution == "positive":
        x = x.abs() + 0.25
    elif distribution == "negative":
        x = -x.abs() - 0.25
    elif distribution == "alternating":
        x[1::2] = -x[::2][: x[1::2].numel()]
    elif distribution == "sparse":
        x.zero_()
        if size:
            indices = torch.randperm(size, device=device, generator=gen)[
                : min(size, 16)
            ]
            x[indices] = torch.randn(indices.numel(), device=device, generator=gen)
    elif distribution != "normal":
        raise ValueError(f"Unknown prefix sum distribution: {distribution}")
    y = torch.empty(size, device=device, dtype=torch.float32).contiguous()
    return x, y


def check_implementation(data: input_t, output: output_t) -> tuple[bool, str]:
    # Parallel reduction orders need an absolute roundoff allowance. Scale it
    # with input energy, not answer magnitude: sqrt(n)-scaled rtol allowed a
    # 16% error on the largest benchmark.
    input_norm = torch.linalg.vector_norm(data[0].to(torch.float64)).item()
    atol = 1e-5 * (1 + input_norm)
    return match_reference(data, output, reference=ref_kernel, rtol=1e-5, atol=atol)
