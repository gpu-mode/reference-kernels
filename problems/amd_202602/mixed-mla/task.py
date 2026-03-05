import torch
from typing import TypeVar, TypedDict, Union

# DeepSeek R1 MLA forward_absorb format:
#
# Input: (q, kv_data, qo_indptr, kv_indptr, config)
#   q:          (total_q, num_heads, qk_head_dim)   bfloat16
#   kv_data:    dict with three KV cache formats:
#     "bf16":   Tensor  (total_kv, 1, 576)           bfloat16
#     "fp8":    (Tensor, Tensor)  kv_buffer fp8 (total_kv, 1, 576) + scalar scale
#     "mxfp4":  (Tensor, Tensor)  kv_buffer fp4x2 (total_kv, 1, 288) + fp8_e8m0 scale
#   qo_indptr:  (batch_size + 1,)                    int32
#   kv_indptr:  (batch_size + 1,)                    int32
#   config:     dict with MLA parameters
#
# where qk_head_dim = kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576
#
# Output: attention output tensor (total_q, num_heads, v_head_dim) bfloat16
#   where v_head_dim = kv_lora_rank = 512
#
# The kv_buffer stores the compressed KV representation:
#   - Full 576 dims used as keys (for Q@K^T score computation)
#   - First 512 dims (kv_lora_rank) used as values (for output computation)

input_t = TypeVar(
    "input_t",
    bound=tuple[torch.Tensor, dict, torch.Tensor, torch.Tensor, dict],
)
output_t = TypeVar("output_t", bound=torch.Tensor)


class TestSpec(TypedDict):
    batchsize: int
    qseqlen: int
    kvseqlen: int
    seed: int
