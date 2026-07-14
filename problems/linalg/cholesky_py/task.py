from typing import NotRequired, TypeVar, TypedDict

import torch


input_t = TypeVar("input_t", bound=torch.Tensor)
output_t = TypeVar("output_t", bound=torch.Tensor)


class TestSpec(TypedDict):
    batch: int
    n: int
    cond: int
    seed: int
    case: NotRequired[str]
