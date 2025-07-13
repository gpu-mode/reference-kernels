from typing import TypedDict, TypeVar
import torch

input_t = TypeVar(
    "input_t", bound=tuple[torch.Tensor, torch.Tensor]
)  # Input is a pair of tensors (input, output) where input is (H, W, 3) RGB tensor and output is (H, W) grayscale tensor
output_t = TypeVar(
    "output_t", bound=torch.Tensor
)  # Output will be (H, W) grayscale tensor


class TestSpec(TypedDict):
    size: int  # Size of the square image (H=W)
    seed: int
