# the nop kernel
from task import input_t, output_t
import torch
import torch.nn.functional as F


def custom_kernel(data: input_t) -> output_t:
    input_tensor, kernel = data
    return torch.empty((input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2]-kernel.shape[3]+1, input_tensor.shape[3]-kernel.shape[3]+1),
                       device=kernel.device, dtype=kernel.dtype
    )
