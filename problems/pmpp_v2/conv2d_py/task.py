from typing import TypedDict, TypeVar, Tuple
import torch

input_t = TypeVar("input_t", bound=Tuple[torch.Tensor, torch.Tensor])
output_t = TypeVar("output_t", bound=torch.Tensor)


class TestSpec(TypedDict):
    size: int
    kernelsize: int
    channels: int
    batch: int
    seed: int   