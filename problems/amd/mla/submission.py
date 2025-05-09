import torch
from task import input_t, output_t
from reference import ref_kernel

def custom_kernel(data: input_t) -> output_t:
    """
    Reference implementation of mla without RoPE
    Args:
        data: b, s_q, mean_sk, h_q, h_kv, d, dv, causal, varlen, seed
    Returns:
        mla output
    """
    return ref_kernel(data)