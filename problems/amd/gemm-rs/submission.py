from task import input_t, output_t
import torch


def custom_kernel(data: input_t) -> output_t:
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