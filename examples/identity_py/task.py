from typing import TypedDict
import torch


input_t = torch.Tensor
output_t = input_t


class TestSpec(TypedDict):
    size: int
    seed: int
