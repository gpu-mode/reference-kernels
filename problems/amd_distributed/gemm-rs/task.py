from typing import TypedDict, TypeVar, Tuple, Optional
import torch

input_t = TypeVar("input_t", bound=Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]])
output_t = TypeVar("output_t", bound=torch.Tensor)


class TestSpec(TypedDict):
    world_size: int
    m: int
    n: int
    k: int
    has_bias: bool
    seed: int
