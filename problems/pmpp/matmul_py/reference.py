import torch
from task import input_t, output_t
from utils import make_match_reference


def generate_input(m: int, n: int, k: int, seed: int) -> input_t:
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    a = torch.empty(m, k, device='cuda', dtype=torch.float16)
    a.uniform_(0, 1, generator=gen)
    b = torch.empty(k, n, device='cuda', dtype=torch.float16)
    b.uniform_(0, 1, generator=gen)
    c = torch.empty(m, n, device='cuda', dtype=torch.float16)
    return (a, b, c)


def ref_kernel(data: input_t) -> output_t:
    a, b = data
    return a @ b


check_implementation = make_match_reference(ref_kernel)
