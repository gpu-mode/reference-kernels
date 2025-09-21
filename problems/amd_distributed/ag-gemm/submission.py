from task import input_t, output_t
import torch


def custom_kernel(data: input_t) -> output_t:
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
