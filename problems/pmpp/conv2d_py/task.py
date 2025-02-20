from typing import TypedDict, TypeVar, Tuple
import torch
from dataclasses import dataclass

input_t = TypeVar("input_t", bound=Tuple[torch.Tensor, torch.Tensor])
output_t = TypeVar("output_t", bound=torch.Tensor)

@dataclass
class KernelSpec:
    stride: int
    padding: int

class TestSpec(TypedDict):
    size: int
    kernel_size: int
    channels: int
    batch: int
    seed: int 