import torch
from typing import TypeVar, TypedDict
from reference import Config, KVCache
input_t = TypeVar("input_t", bound=tuple[Config, torch.Tensor, KVCache])
output_t = TypeVar("output_t", bound=torch.Tensor)

# Define test spec with parameters in the same order as in task.yml
class TestSpec(TypedDict):
    bs: int     # batch size
    sq: int     # query sequence length
    sk: int     # kv sequence length
    dim: int    # hidden size
    seed: int   # random seed