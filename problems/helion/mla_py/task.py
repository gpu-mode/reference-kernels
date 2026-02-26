from typing import TypedDict, TypeVar
import torch

input_t = TypeVar("input_t", bound=tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor])
output_t = TypeVar("output_t", bound=torch.Tensor)

class TestSpec(TypedDict):
    B: int
    H: int
    S: int
    d_c: int
    d_r: int
    seed: int
