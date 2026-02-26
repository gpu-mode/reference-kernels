from typing import TypedDict, TypeVar
import torch

input_t = TypeVar("input_t", bound=tuple[torch.Tensor, torch.Tensor])
output_t = TypeVar("output_t", bound=torch.Tensor)

class TestSpec(TypedDict):
    batch_size: int
    vocab_size: int
    seed: int
