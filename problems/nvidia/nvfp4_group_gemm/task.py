import torch
from typing import TypedDict, TypeVar

input_t = TypeVar("input_t", bound=tuple[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]], list[tuple[int, int, int, int]]])
output_t = TypeVar("output_t", bound=list[torch.Tensor])
class TestSpec(TypedDict):
    problem_sizes: list[tuple[int, int, int, int]]
    seed: int