import torch
from typing import TypeVar, TypedDict

input_t = TypeVar("input_t", bound=tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor])
output_t = TypeVar("output_t", bound=torch.Tensor)

# Define test spec with parameters in the same order as in task.yml
class TestSpec(TypedDict):
    b: int      # batch size
    d: int      # dimension
    dv: int     # value dimension
    hq: int     # number of query heads
    sq: int     # query sequence length
    hkv: int    # number of key/value heads
    meansk: int # mean kv sequence length
    seed: int   # random seed