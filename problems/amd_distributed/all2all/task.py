import torch
from typing import TypeVar, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    from reference import MoEConfig, RankTestData


input_t = TypeVar("input_t", bound=tuple["MoEConfig", "RankTestData", int, int])
output_t = TypeVar("output_t", bound=torch.Tensor)


class TestSpec(TypedDict):
    num_experts: int
    experts_per_token: int
    hidden_dim: int
    max_num_tokens: int
    seed: int
