from utils import make_match_reference
from task import input_t, output_t
import torch
from typing import Optional
import os
import datetime


def generate_input(RANK: int, world_size: int, m: int, n: int, k: int, seed: int) -> input_t:
    """
    Generate random input and weights for the AG-GEMM operation.

    Returns:
        Tuple of (
            input: torch.Tensor,  # [M, local_K]
            weight: torch.Tensor,  # [N, local_K]
            transposed_weight: bool,  # Whether the weight is transposed
            bias: Optional[torch.Tensor],  # [N] or None
        )
    """
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed + RANK)

    assert m % world_size == 0, "m must be divisible by world_size"
    assert k % world_size == 0, "k must be divisible by world_size"
    local_k = k // world_size

    # Generate random inputs and weights
    input = (torch.rand((m, local_k), dtype=torch.float16, device="cuda", generator=gen) * 2 - 1) * 0.01
    weight = (torch.rand((n, local_k), dtype=torch.float16, device="cuda", generator=gen) * 2 - 1) * 0.01

    return (input, weight, False, None)


def ref_kernel(data: input_t) -> output_t:
    """
    Reference kernel for Gemm-ReduceScatter operation.

    Args:
        data: Tuple of (input: torch.Tensor, weight: torch.Tensor, transposed_weight: bool,
                bias: Optional[torch.Tensor])
            - input: Local input tensor of shape [M, local_K].
            - weight: Weight tensor of shape [N, local_K] or [local_K, N] if transed_weight is True.
            - transposed_weight: Whether the weight is transposed.
            - bias: Optional bias tensor of shape [N] or None.
    Returns:
        Tuple containing:
            - output: Resulting tensor of shape [M // world_size, N].
    """
    input, weight, transposed_weight, bias = data
    M, local_K = input.shape
    if not transposed_weight:
        weight = weight.T
    N = weight.shape[1]
    world_size = torch.distributed.get_world_size()
    # matmul
    output = torch.matmul(input, weight)
    if bias is not None:
        output = output + bias
    # reduce scatter
    rs_output = torch.empty((M // world_size, N), dtype=output.dtype, device=input.device)
    torch.distributed.reduce_scatter_tensor(rs_output, output)
    return rs_output


check_implementation = make_match_reference(ref_kernel, rtol=1e-2, atol=1e-2)


#
# TEST CODE FOR REFERENCE
#

# import os
# import datetime
# # Initialize distributed environment
# RANK = int(os.environ.get("RANK", 0))
# LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
# WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
# torch.cuda.set_device(LOCAL_RANK)
# torch.distributed.init_process_group(
#     backend="nccl",
#     world_size=WORLD_SIZE,
#     rank=RANK,
#     timeout=datetime.timedelta(seconds=1800),
# )

# # Generate input data
# world_size = WORLD_SIZE
# m = 8192
# n = 8192
# k = 29568
# seed = 42
# input_datas = generate_input(RANK, world_size, m, n, k, seed)

# # Run the reference kernel
# import time
# torch.cuda.synchronize()
# start_time = time.time()
# output = ref_kernel(*input_datas)
# torch.cuda.synchronize()
# end_time = time.time()
# elapsed_ms = (end_time - start_time) * 1000
# if RANK == 0:
#     print(f"Reference kernel time: {elapsed_ms:.2f} ms")

# # Destroy process group
# torch.distributed.destroy_process_group()