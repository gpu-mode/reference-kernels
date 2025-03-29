import torch
from task import input_t, output_t
from utils import make_match_reference


block_shape = (128, 128)

def generate_input(m: int, n: int, k: int, seed: int) -> input_t:
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    block_shape_n, block_shape_k = block_shape
    scale_n =  (n + block_shape_n - 1) // block_shape_n
    scale_k =  (k + block_shape_k - 1) // block_shape_k

    # Generate random inputs with FP8 quantization
    a = (torch.randn((m, k), dtype=torch.float16, generator=gen)/10).to(torch.float8_e4m3fnuz)
    b = (torch.randn((n, k), dtype=torch.float16, generator=gen)/10).to(torch.float8_e4m3fnuz)

    # Generate scaling factors
    a_scale = torch.randn([m, scale_k], dtype=torch.float32, generator=gen)
    b_scale = torch.randn([scale_n, scale_k], dtype=torch.float32, generator=gen)

    # Apply scaling to input 'a'
    a = a.to(a_scale.dtype).view(m, k//block_shape_k, block_shape_k) * a_scale.unsqueeze(-1)
    a = a.view(m, k)

    # Apply scaling to input 'b'
    b_scale = (
        b_scale.view(-1, 1)
        .repeat(1, block_shape_n * block_shape_k)
        .view(scale_n, scale_k, block_shape_n, block_shape_k)
        .permute(0, 2, 1, 3)  # Reorder dimensions: [scale_n, blk_n, scale_k, blk_k]
        .reshape(scale_n * block_shape_n, scale_k * block_shape_k)
    )
    b_scale = b_scale[:n, :k]
    b = b.to(b_scale.dtype) * b_scale
    return (a, b)


def ref_kernel(data: input_t) -> output_t:
    a, b, = data
    return a @ b


check_implementation = make_match_reference(ref_kernel)

