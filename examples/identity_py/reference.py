import torch
import sys
from task import input_t, output_t, N_SIZES, Ns
from utils import verbose_allclose

def generate_input() -> input_t:
    data = [torch.empty(N) for N in Ns]

    for i in range(N_SIZES):
        data[i].uniform_(0, 1)
    return data

def ref_kernel(data: input_t) -> output_t:
    return data


def check_implementation(submission_output, reference_output) -> bool:
    for i in range(N_SIZES):
        reasons = verbose_allclose(submission_output[i], reference_output[i])
        if len(reasons) > 0:
            print("mismatch found! custom implementation doesnt match reference.", file=sys.stderr)
            for reason in reasons:
                print(reason, file=sys.stderr)
            return False
    
    return True


