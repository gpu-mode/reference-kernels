from task import input_t, output_t
import torch


def generate_input(rank: int, world_size: int, m: int, n: int, k: int, has_bias: bool, seed: int) -> input_t:
    """
    Generate random input and weights for the Allgather-Gemm operation.

    Returns:
        Tuple of (
            input: torch.Tensor,  # [local_M, k]
            weight: torch.Tensor,  # [local_N, K]
            bias: Optional[torch.Tensor],  # [local_N] or None
        )
    """
    device = torch.device(f"cuda:{rank}")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + rank)

    assert m % world_size == 0, "m must be divisible by world_size"
    assert n % world_size == 0, "n must be divisible by world_size"
    local_m = m // world_size
    local_n = n // world_size

    # Generate random inputs and weights
    input = (torch.rand((local_m, k), dtype=torch.bfloat16, device=device, generator=gen) * 2 - 1) * 0.01
    weight = (torch.rand((local_n, k), dtype=torch.bfloat16, device=device, generator=gen) * 2 - 1) * 0.01

    bias = None
    if has_bias:
        bias = (torch.rand((local_n,), dtype=torch.bfloat16, device=device, generator=gen) * 2 - 1) * 0.01
    return (input, weight, bias)


def ref_kernel(data: input_t) -> output_t:
    """
    Reference kernel for AG-GEMM operation.
    Args:
        data: Tuple of (input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor])
            - input: Local input tensor of shape [local_M, K].
            - weight: Weight tensor of shape [local_N, K].
            - bias: Optional bias tensor of shape [local_N] or None.
    Returns:
        output: Resulting tensor of shape [local_M * world_size, local_N].
    """
    input, weight, bias = data
    local_M, K = input.shape
    world_size = torch.distributed.get_world_size()
    full_input = torch.empty((local_M * world_size, K), dtype=input.dtype, device=input.device)
    # allgather
    torch.distributed.all_gather_into_tensor(full_input, input)
    # matmul
    output = torch.matmul(full_input, weight.T)

    if bias is not None:
        output = output + bias

    return output


def check_implementation(data: input_t, output: output_t):
    expected = ref_kernel(data)
    if output.device != expected.device:
        return False, f"Output device mismatch: {output.device} != {expected.device}"
    res = torch.allclose(output, expected, rtol=1e-2, atol=1e-2)
    if not res:
        return False, f"Output values mismatch, {output} != {expected}"

    return True, ""
