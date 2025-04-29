from typing import TypedDict, TypeVar, Tuple, Dict
import torch

input_t = TypeVar("input_t", bound=Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict])
output_t = TypeVar("output_t", bound=Tuple[torch.Tensor, Dict])


class TestSpec(TypedDict):
    d_hidden: int
    d_expert: int
    n_routed_experts: int
    n_shared_experts: int
    n_experts_per_token: int
    batch_size: int
    seq_len: int
    seed: int