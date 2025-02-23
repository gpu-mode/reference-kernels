from task import input_t, output_t
import torch
import torch.nn.functional as F


def custom_kernel(data: input_t) -> output_t:
    input_tensor, kernel = data
    return F.conv2d(
        input_tensor,
        kernel,
        stride=1,
        padding=0
    )
