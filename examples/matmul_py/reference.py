import torch
from task import input_t, output_t
from utils import verbose_allclose

def generate_input(m: int, n: int, k: int, seed: int) -> input_t:
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    a = torch.empty(m, k, device='cuda', dtype=torch.float16)
    a.uniform_(0, 1, generator=gen)
    b = torch.empty(k, n, device='cuda', dtype=torch.float16)
    b.uniform_(0, 1, generator=gen)
    return (a, b)

def ref_kernel(data: input_t) -> output_t:
    a, b = data
    return a @ b

def check_implementation(data: input_t, output: output_t) -> str:
    expected = ref_kernel(data)
    reasons = verbose_allclose(output, expected)
    if len(reasons) > 0:
        # TODO better processing of reasons
        return "mismatch found! custom implementation doesn't match reference.: " + reasons[0]

    return ''

