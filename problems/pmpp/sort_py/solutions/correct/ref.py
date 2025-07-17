import torch
from task import input_t, output_t


def _custom_kernel(data: input_t) -> output_t:
    data, output = data
    output = torch.sort(data)[0]
    return output


custom_kernel = torch.compile(_custom_kernel, mode="reduce-overhead")
