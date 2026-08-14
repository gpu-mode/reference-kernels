import os
import torch
from torch.utils.cpp_extension import load
from task import input_t, output_t

curr_dir = os.path.dirname(os.path.abspath(__file__))
mla_hip = load(
    name="mla_kernel",
    sources=[os.path.join(curr_dir, "mla_kernel.hip")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--offload-arch=gfx942"],
    verbose=False,
)

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config["batch_size"]
    num_heads = config["num_heads"]
    q_seq_len = config["q_seq_len"]
    kv_seq_len = config["kv_seq_len"]
    sm_scale = config["sm_scale"]

    total_q = q.shape[0]

    q_nope = q[:, :, :KV_LORA_RANK].reshape(batch_size, q_seq_len, num_heads, KV_LORA_RANK).contiguous()
    q_pe = q[:, :, KV_LORA_RANK:].reshape(batch_size, q_seq_len, num_heads, QK_ROPE_HEAD_DIM).contiguous()

    kv_fp4, kv_scale_e8m0 = kv_data["mxfp4"]

    total_kv = kv_fp4.shape[0]
    kv_cache = kv_fp4.view(batch_size, kv_seq_len, 288).contiguous()
    if kv_cache.dtype != torch.uint8:
        kv_cache = kv_cache.view(torch.uint8)

    num_scale_blocks = kv_scale_e8m0.shape[-1]
    kv_scale = kv_scale_e8m0.reshape(batch_size, kv_seq_len, num_scale_blocks)[:, :, :18].contiguous()
    if kv_scale.dtype != torch.uint8:
        kv_scale = kv_scale.view(torch.uint8)

    out = mla_hip.forward(
        q_nope.to(torch.bfloat16),
        q_pe.to(torch.bfloat16),
        kv_cache,
        kv_scale,
        float(sm_scale),
    )

    out = out.view(total_q, num_heads, KV_LORA_RANK)
    return out
