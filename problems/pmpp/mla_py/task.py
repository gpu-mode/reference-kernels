import torch
from typing import TypeVar, TypedDict

input_t = TypeVar("input_t", bound=tuple[torch.Tensor, torch.Tensor, torch.Tensor])
output_t = TypeVar("output_t", bound=torch.Tensor)

class TestSpec(TypedDict):
    B: int
    S: int
    H_Q: int
    H_KV: int
    D: int
    D_V: int
    seed: int