from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    A, B, output = data
    output[...] = A + B
    return output
