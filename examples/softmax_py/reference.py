from typing import List

import torch


def check_implementation(custom_output, ref_output) -> bool:
    for c, r in zip(custom_output, ref_output, strict=False):
        if not torch.allclose(c, r, atol=1e-5):
            print("mismatch found! custom implementation doesnt match reference.")
            return False

    return True


def ref_kernel(xs: List[torch.Tensor], dim: int = -1) -> torch.Tensor:
    """
    Reference implementation of the Softmax function using PyTorch's predefined functions.
    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension along which to apply softmax.
    Returns:
        torch.Tensor: Tensor after applying Softmax.
    """

    return [torch.nn.functional.softmax(x, dim=dim) for x in xs]


def generate_input(seed: int = None, to_cuda: bool = True) -> List[torch.Tensor]:
    """
    Generates random input tensor of the specified shape.
    Args:
        seed (int): Random seed for reproducibility.
        to_cuda (bool): Whether to use GPU (CUDA or ROCm) or CPU.
    Returns:
        List[torch.Tensor]: List of randomly generated tensors.
    """
    shapes = [(128, 64), (256, 64), (512, 64)]

    # Determine the device
    if to_cuda:
        if torch.cuda.is_available():  # Check for NVIDIA GPU
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():  # Check for AMD GPU using MPS backend
            device = torch.device("mps")
        else:
            print("No compatible GPU found. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    if seed is not None:
        torch.manual_seed(seed)

    tensors = []
    for shape in shapes:
        tensors.append(torch.randn(shape, device=device))

    return tensors


if __name__ == "__main__":
    inputs = generate_input(seed=42)
    for idx, tensor in enumerate(inputs):
        print(f"Input Tensor {idx + 1} (Shape: {tensor.shape}):\n{tensor}")
