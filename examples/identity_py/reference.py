from typing import List

import torch


def check_implementation(custom_output: List[torch.Tensor], ref_output: List[torch.Tensor]) -> bool:
    for c, r in zip(custom_output, ref_output, strict=False):
        if not torch.allclose(c, r, atol=1e-5):
            print("mismatch found! custom implementation doesnt match reference.")
            return False

    return True


def ref_kernel(xs: List[torch.Tensor]) -> List[torch.Tensor]:
    return xs


def generate_input() -> List[torch.Tensor]:
    """
    Generates random input tensor of the specified shape.
    Returns:
        List[torch.Tensor]: List of randomly generated tensors.
    """
    shapes = [(128, 64), (256, 64), (512, 64)]

    # Determine the device
    if torch.cuda.is_available():  # Check for NVIDIA GPU
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # Check for AMD GPU using MPS backend
        device = torch.device("mps")
    else:
        print("No compatible GPU found. Falling back to CPU.")
        device = torch.device("cpu")

    tensors = []
    for shape in shapes:
        tensors.append(torch.randn(shape, device=device))

    return tensors


if __name__ == "__main__":
    inputs = generate_input(seed=42)
    for idx, tensor in enumerate(inputs):
        print(f"Input Tensor {idx + 1} (Shape: {tensor.shape}):\n{tensor}")
