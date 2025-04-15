#!POPCORN leaderboard amd-identity
from task import input_t, output_t
import torch


def custom_kernel(data: input_t) -> output_t:
    input, output = data
    output[...] = input
    return output
