# the nop kernel
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    _, _, output = data
    return output
