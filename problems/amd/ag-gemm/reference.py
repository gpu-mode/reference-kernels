from utils import make_match_reference
from task import input_t, output_t
import torch


def generate_input(RANK: int, world_size: int, m: int, n: int, k: int, seed: int) -> input_t:
    """
    Generate random input and weights for the AG-GEMM operation.

    Returns:
        Tuple of (
            input: torch.Tensor,  # [local_M, k]
            weight: torch.Tensor,  # [local_N, K]
            transposed_weight: bool,  # Whether the weight is transposed
            bias: Optional[torch.Tensor],  # [local_N] or None
        )
    """
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed + RANK)

    assert m % world_size == 0, "m must be divisible by world_size"
    assert n % world_size == 0, "n must be divisible by world_size"
    local_m = m // world_size
    local_n = n // world_size

    # Generate random inputs and weights
    input = (torch.randn((local_m, k), dtype=torch.float16, device="cuda", generator=gen) * 2 - 1) * 0.01
    weight = (torch.randn((local_n, k), dtype=torch.float16, device="cuda", generator=gen) * 2 - 1) * 0.01

    return (input, weight, False, None)


def ref_kernel(data: input_t) -> output_t:
    """
    Reference kernel for AG-GEMM operation using PyTorch.

    Args:
        data: Tuple of (input: torch.Tensor, weights: torch.Tensor, transposed_weight: bool,
                bias: Optional, None or torch.Tensor).
            - input: Local input tensor of shape [local_M, K].
            - weight: Weight tensor of shape [local_N, K] or [K, local_N] if transed_weight is True.
            - transposed_weight: Whether the weight is transposed.
            - bias: bias tensor of shape [local_N] or None.
    Returns:
        Tuple containing:
            - output: Resulting tensor of shape [local_M * world_size, local_N].
    """
    input, weight, transposed_weight, bias = data
    local_M, K = input.shape
    world_size = torch.distributed.get_world_size()
    if transposed_weight:
        assert K == weight.shape[0]
    else:
        assert K == weight.shape[1]
        weight = weight.T
    # allgather input across all ranks
    full_input = torch.empty((local_M * world_size, K), dtype=input.dtype, device=input.device)
    torch.distributed.all_gather_into_tensor(full_input, input)
    # matmul
    output = torch.matmul(full_input, weight)

    if bias is not None:
        output = output + bias

    return output


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
# start_time = time.perf_counter()
# output = ref_kernel(*input_datas)
# torch.cuda.synchronize()
# end_time = time.perf_counter()
# elapsed_ms = (end_time - start_time) * 1000
# if RANK == 0:
#     print(f"Reference kernel time: {elapsed_ms:.2f} ms")

# # Destroy process group
# torch.distributed.destroy_process_group()