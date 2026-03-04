from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    import torch
    import torch.nn.functional as F

    x, weight, bias = data
    W = weight.shape[1]
    D = x.shape[1]

    x_padded = F.pad(x, (W - 1, 0))
    output = F.conv1d(x_padded, weight.unsqueeze(1), bias=bias, groups=D)
    return output
