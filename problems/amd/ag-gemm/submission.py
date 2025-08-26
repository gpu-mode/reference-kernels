from task import input_t, output_t
import torch


def custom_kernel(data: input_t) -> output_t:
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