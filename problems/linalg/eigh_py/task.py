import torch
from typing import NotRequired, TypeVar, TypedDict

input_t = TypeVar("input_t", bound=torch.Tensor)
output_t = TypeVar("output_t", bound=tuple[torch.Tensor, torch.Tensor])


class TestSpec(TypedDict):
    batch: int
    n: int
    cond: int
    seed: int
    case: NotRequired[str]
