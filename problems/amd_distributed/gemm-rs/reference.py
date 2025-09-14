from task import input_t, output_t
import torch


def generate_input(rank: int, world_size: int, m: int, n: int, k: int, has_bias: bool, seed: int) -> input_t:
    """
    Generate random input and weights for the Gemm-ReduceScatter operation.

    Returns:
        Tuple of (
            input: torch.Tensor,  # [M, local_K]
            weight: torch.Tensor,  # [N, local_K]
            bias: Optional[torch.Tensor],  # [N] or None
        )
    """
    device = torch.device(f'cuda:{rank}')
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + rank)

    assert m % world_size == 0, "m must be divisible by world_size"
    assert k % world_size == 0, "k must be divisible by world_size"
    local_k = k // world_size

    # Generate random inputs and weights
    input = (torch.rand((m, local_k), dtype=torch.bfloat16, device=device, generator=gen) * 2 - 1) * 0.01
    weight = (torch.rand((n, local_k), dtype=torch.bfloat16, device=device, generator=gen) * 2 - 1) * 0.01

    bias = None
    if has_bias:
        gen.manual_seed(seed)
        bias = (torch.rand((n,), dtype=torch.bfloat16, device=device, generator=gen) * 2 - 1) * 0.01

    return (input, weight, bias)


def ref_kernel(data: input_t) -> output_t:
    """
    Reference kernel for Gemm-ReduceScatter operation.

    Args:
        data: Tuple of (input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor])
            - input: Local input tensor of shape [M, local_K].
            - weight: Weight tensor of shape [N, local_K].
            - bias: Optional bias tensor of shape [N] or None.
    Returns:
        Tuple containing:
            - output: Resulting tensor of shape [M // world_size, N].
    """
    input, weight, bias = data
    M, local_K = input.shape
    N = weight.shape[0]
    world_size = torch.distributed.get_world_size()
    # matmul
    output = torch.matmul(input, weight.T)
    if bias is not None:
        output = output + bias
    # reduce scatter
    rs_output = torch.empty((M // world_size, N), dtype=output.dtype, device=input.device)
    torch.distributed.reduce_scatter_tensor(rs_output, output)
    return rs_output


def check_implementation(data: input_t, output: output_t):
    expected = ref_kernel(data)
    if output.device != expected.device:
        return False, f"Output device mismatch: {output.device} != {expected.device}"
    res = torch.allclose(output, expected, rtol=1e-2, atol=1e-2)
    if not res:
        return False, f"Output values mismatch, {output} != {expected}"

    return True, ""
