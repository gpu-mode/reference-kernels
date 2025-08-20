import torch
from typing import TypeVar, TypedDict

input_t = TypeVar("input_t", bound=tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor])
output_t = TypeVar("output_t", bound=torch.Tensor)

class TestSpec(TypedDict):
    num_experts: int
    experts_per_token: int
    hidden_dim: int
    max_num_tokens: int
    seed: int