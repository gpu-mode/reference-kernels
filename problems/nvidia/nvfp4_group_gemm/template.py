from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """
    Reference implementation of block-scale fp4 group gemm
    Args:
        data: list of tuples (abc_tensors, sfasfb_tensors, problem_sizes) where:
            abc_tensors: list of tuples (a, b, c) where 
                a is torch.Tensor[float4e2m1fn_x2] of shape [m, k // 2, l]
                b is torch.Tensor[float4e2m1fn_x2] of shape [n, k // 2, l]
                c is torch.Tensor[float16] of shape [m, n, l]
            sfasfb_tensors: list of tuples (sfa, sfb) where 
                sfa is torch.Tensor[float8_e4m3fnuz] of shape [m, k // 16, l]
                sfb is torch.Tensor[float8_e4m3fnuz] of shape [n, k // 16, l]
            problem_sizes: list of tuples (m, n, k, l)
        each group has its own a, b, c, sfa, sfb with different m, n, k, l problem sizes
        l should always be 1 for each group.
    Returns:
        list of tuples (c) where c is torch.Tensor[float16] of shape [m, n, l]
    """
    abc_tensors, sfasfb_tensors, problem_sizes = data
    result_tensors = []
    for i, (a, b, c) in enumerate(abc_tensors):
        # add you implementation here
        result_tensors.append(c)

    return result_tensors