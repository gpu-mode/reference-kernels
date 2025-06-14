import torch
from utils import verbose_allclose, get_device
from task import input_t, output_t
from typing import Tuple


def generate_input(size: int, seed: int) -> input_t:
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    data = torch.empty(size, device='cuda', dtype=torch.float16)
    data.uniform_(0, 1, generator=gen)
    return data

def ref_kernel(data: input_t) -> output_t:
    return torch.nn.functional.softmax(data, dim=-1)

def check_implementation(data: input_t, output: output_t) -> Tuple[bool, str]:

    expected = ref_kernel(data)
    reasons = verbose_allclose(output, expected)

    if len(reasons) > 0:
        return False, "mismatch found! custom implementation doesn't match reference: " + reasons[0]

    return True, ''
