from typing import TypedDict, TypeVar
import torch

input_t = TypeVar("input_t", bound=torch.Tensor)  # Input will be (H, W, 3) RGB tensor
output_t = TypeVar("output_t", bound=torch.Tensor)  # Output will be (H, W) grayscale tensor

class TestSpec(TypedDict):
    size: int  # Size of the square image (H=W)
    seed: int 