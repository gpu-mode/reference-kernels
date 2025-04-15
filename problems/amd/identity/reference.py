import torch
from task import input_t, output_t
from utils import make_match_reference


def generate_input(size: int, seed: int) -> input_t:
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    data = torch.empty(size, device='cuda', dtype=torch.float16)
    data.uniform_(0, 1, generator=gen)
    return data, torch.empty_like(data)


def ref_kernel(data: input_t) -> output_t:
    input, output = data
    output[...] = input
    return output


check_implementation = make_match_reference(ref_kernel)
