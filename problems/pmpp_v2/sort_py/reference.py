import torch
from task import input_t, output_t
from utils import DeterministicContext, make_match_reference_exact


def ref_kernel(data: input_t) -> output_t:
    """
    Reference implementation of sort using PyTorch.
    Args:
        data: Input tensor to be sorted
    Returns:
        Sorted tensor
    """
    with DeterministicContext():
        data, output = data
        output[...] = torch.sort(data)[0]
        return output


def generate_input(
    size: int, seed: int, distribution: str = "normal", *, device="cuda"
) -> input_t:
    """Seeds select random values, never their magnitude or location.

    Private seeds can be very large. Adding them to float32 data erases the
    random values and makes approximate, input-independent outputs pass.
    """
    if size < 0:
        raise ValueError("size must be non-negative")
    gen = torch.Generator(device=device).manual_seed(seed)
    if distribution in ("normal", "sorted", "reverse", "outliers"):
        values = torch.randn(size, device=device, dtype=torch.float32, generator=gen)
        if distribution == "sorted":
            values = values.sort().values
        elif distribution == "reverse":
            values = values.sort(descending=True).values
        elif distribution == "outliers" and size:
            indices = torch.randperm(size, device=device, generator=gen)[
                : max(1, size // 32)
            ]
            values[indices] *= 1024
    elif distribution == "uniform":
        values = (
            torch.rand(size, device=device, dtype=torch.float32, generator=gen) * 16 - 8
        )
    elif distribution == "duplicates":
        values = torch.randint(-8, 9, (size,), device=device, generator=gen).to(
            torch.float32
        )
    elif distribution == "constant":
        value = torch.randint(-8, 9, (), device=device, generator=gen).item()
        values = torch.full((size,), value, device=device, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown sort distribution: {distribution}")
    values = values.contiguous()
    return values, torch.empty_like(values)


# Sorting preserves values exactly. Tolerance accepts fabricated values and
# lost duplicates even when the output is increasing and looks plausible.
check_implementation = make_match_reference_exact(ref_kernel)
