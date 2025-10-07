import torch
from task import input_t, output_t
from utils import make_match_reference


def ref_kernel(
    data: input_t,
)->output_t:
    a, b, c = data
    # call torch matmul operation
    c[...] = a @ b
    return c


def generate_input(
    m: int,
    n: int,
    k: int,
    seed: int,
) -> input_t:
    torch.manual_seed(seed)

    # Generate a, b and c tensors
    a = torch.empty(m, k, dtype=torch.float16).random_(-2, 2).to(device="cuda")
    b = torch.empty(n, k, dtype=torch.float16).random_(-2, 2).to(device="cuda")
    c = torch.empty(m, n, dtype=torch.float16).random_(-2, 2).to(device="cuda")

    return (a, b, c)


check_implementation = make_match_reference(ref_kernel, rtol=1e-01, atol=1e-05)