from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """
    Reference implementation of block-scale fp8 gemm
    Args:
        data: Tuple that expands to:
            a: torch.Tensor[float16] of shape [m, k],
            b: torch.Tensor[float16] of shape [n, k],
            c: torch.Tensor[float16] of shape [m, n]
    Returns:
        Tensor containing output in float16
    """
    # c: [m, n] is pre-allocated memory to avoid timing allocation overhead.
    a, b, c = data

    # Your implementation here

    return c
