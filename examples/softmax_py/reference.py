from typing import List
import torch
from utils import get_device


def ref_kernel(xs: List[torch.Tensor], dim: int = -1) -> List[torch.Tensor]:
    """
    Reference implementation of the Softmax function using PyTorch's predefined functions.
    Args:
        xs (List[torch.Tensor]): List of input tensors.
        dim (int): Dimension along which to apply softmax.
    Returns:
        List[torch.Tensor]: List of tensors after applying Softmax.
    """
    return [torch.nn.functional.softmax(x, dim=dim) for x in xs]


def check_implementation(custom_output, ref_output) -> bool:
    """Check if custom implementation matches reference implementation."""
    for c, r in zip(custom_output, ref_output, strict=False):
        if not torch.allclose(c, r, atol=1e-5):
            print("Mismatch found! Custom implementation doesn't match reference.")
            return False
    return True


def generate_input(seed: int = None, to_cuda: bool = True) -> List[torch.Tensor]:
    """
    Generates random input tensors of specified shapes.
    Args:
        seed (int): Random seed for reproducibility.
        to_cuda (bool): Whether to use GPU or CPU.
    Returns:
        List[torch.Tensor]: List of randomly generated tensors.
    """
    shapes = [(128, 64), (256, 64), (512, 64)]
    device = get_device(to_cuda)

    if seed is not None:
        torch.manual_seed(seed)

    return [torch.randn(shape, device=device) for shape in shapes]


if __name__ == "__main__":
    inputs = generate_input(seed=42)
    for idx, tensor in enumerate(inputs):
        print(f"Input Tensor {idx + 1} (Shape: {tensor.shape}):\n{tensor}")
