from typing import TypeVar, Tuple, Dict
import torch

input_t = TypeVar("input_t", bound=Tuple[
    torch.Tensor,                # hidden_states                   [M, d_hidden]
    torch.Tensor,                # gate_up_weight                  [E, 2*d_expert_pad, d_hidden_pad//2]   fp4x2 (raw)
    torch.Tensor,                # down_weight                     [E, d_hidden_pad, d_expert_pad//2]     fp4x2 (raw)
    torch.Tensor,                # gate_up_weight_scale            [E, 2*d_expert_pad, scale_K]           e8m0  (raw)
    torch.Tensor,                # down_weight_scale               [E, d_hidden_pad, scale_K]             e8m0  (raw)
    torch.Tensor,                # gate_up_weight_shuffled         [E, 2*d_expert_pad, d_hidden_pad//2]   fp4x2 (shuffled)
    torch.Tensor,                # down_weight_shuffled            [E, d_hidden_pad, d_expert_pad//2]     fp4x2 (shuffled)
    torch.Tensor,                # gate_up_weight_scale_shuffled   [padded, flat]                         e8m0  (shuffled)
    torch.Tensor,                # down_weight_scale_shuffled      [padded, flat]                         e8m0  (shuffled)
    torch.Tensor,                # topk_weights                    [M, total_top_k]
    torch.Tensor,                # topk_ids                        [M, total_top_k]
    Dict,                        # config
])
output_t = TypeVar("output_t", bound=torch.Tensor)


class TestSpec:
    dhidden: int          # hidden dimension (7168 for DeepSeek-R1)
    dexpert: int          # intermediate dimension per expert (per partition)
    nroutedexperts: int   # number of local routed experts on this GPU
    nexpertspertoken: int # top-k routed experts per token (8 for DeepSeek-R1)
    nsharedexperts: int   # number of shared experts (1 for DeepSeek-R1), always selected
    bs: int               # batch size = number of tokens in this batch
    seed: int
