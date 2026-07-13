import torch

from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    batch, n, _ = data.shape
    del batch
    block = 1024
    workspace = data.clone()
    output = torch.zeros_like(data)

    for start in range(0, n, block):
        end = min(start + block, n)
        factor = torch.linalg.cholesky_ex(
            workspace[:, start:end, start:end], check_errors=False
        ).L
        output[:, start:end, start:end].copy_(factor)
        if end == n:
            continue
        solved = torch.linalg.solve_triangular(
            factor,
            workspace[:, end:, start:end].transpose(-1, -2),
            upper=False,
            left=True,
        ).transpose(-1, -2)
        output[:, end:, start:end].copy_(solved)
        workspace[:, end:, end:].sub_(solved @ solved.transpose(-1, -2))
    return output
