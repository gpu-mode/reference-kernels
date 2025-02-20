from typing import TypedDict, TypeVar
import torch
from dataclasses import dataclass

input_t = TypeVar("input_t", bound=torch.Tensor)
output_t = TypeVar("output_t", bound=torch.Tensor)

@dataclass
class HistogramSpec:
    num_bins: int
    min_val: float
    max_val: float

class TestSpec(TypedDict):
    size: int
    seed: int 