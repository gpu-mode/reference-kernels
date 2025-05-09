import torch
from task import input_t, output_t
from reference import ref_kernel

def custom_kernel(data: input_t) -> output_t:
    """
    Reference implementation of rope + mla 
    Args:
        data: q, k_cache, v_cache, block_table, cached_seqlens, max_seqlen_pad, rope_positions
    Returns:
        mla output
    """
    return ref_kernel(data)