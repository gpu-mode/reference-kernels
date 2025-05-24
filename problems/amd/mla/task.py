import torch
from typing import TypeVar, TypedDict

input_t = TypeVar("input_t", bound=tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int])
output_t = TypeVar("output_t", bound=tuple[torch.Tensor, torch.Tensor])

class TestSpec(TypedDict):
    b: int
    s_q: int
    mean_sk: int
    h_q: int
    h_kv: int
    d: int
    dv: int
    causal: bool
    var_len: bool
    seed: int