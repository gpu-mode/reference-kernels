from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """
    Copies the contents of `input` into `output`
    Args:
        data: tuple of (input, output) tensors

    Returns: output tensor
    """
    input, output = data
    # implement processing
    return output
