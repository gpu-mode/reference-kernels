import argparse
import torch
from typing import Tuple

import cutlass
import cutlass.cute as cute

input_t = TypeVar("input_t", bound=tuple[cute.Tensor, cute.Tensor, cute.Tensor])
output_t = TypeVar("output_t", bound=cute.Tensor)

class TestSpec(TypedDict):
    m: int
    n: int
    k: int
    seed: int