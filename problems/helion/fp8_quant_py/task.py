from typing import TypedDict, TypeVar
import torch

input_t = TypeVar("input_t", bound=tuple[torch.Tensor, torch.Tensor, torch.Tensor])
output_t = TypeVar("output_t", bound=tuple[torch.Tensor, torch.Tensor])

class TestSpec(TypedDict):
    num_tokens: int
    hidden_dim: int
    group_size: int
    seed: int
