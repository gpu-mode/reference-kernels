from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """
    Reference implementation of block-scale fp8 gemm
    Args:
        data: Tuple that expands to:
            a: torch.Tensor[float4e2m1fn] of shape [l, m, k],
            b: torch.Tensor[float4e2m1fn] of shape [l, 1, k],
            scale_a: torch.Tensor[float8_e4m3fnuz] of shape [l, m, k // 16],
            scale_b: torch.Tensor[float8_e4m3fnuz] of shape [l, 1, k // 16],
            c: torch.Tensor[float16] of shape [l, m, 1]
    Returns:
        Tensor containing output in float16
        c: torch.Tensor[float16] of shape [l, m, 1]
    """
    # c: [l, m, 1] is pre-allocated memory to avoid timing allocation overhead.
    a, b, scale_a, scale_b, c = data

    # Your implementation here

    return c