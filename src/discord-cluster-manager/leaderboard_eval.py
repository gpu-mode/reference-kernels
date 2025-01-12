########
# Evaluation scripts to run for leaderboard results
########

from pathlib import Path

py_eval = """
import torch
import time
from reference import ref_kernel, generate_input, check_implementation
from train import custom_kernel


def correctness() -> bool:
    for _ in range(10):  # check multiple times
        inputs = generate_input()

        custom_output = custom_kernel(inputs)
        ref_output = ref_kernel(inputs)

        if not check_implementation(custom_output, ref_output):
            return False

    print('custom implementation matches the reference implementation.')
    return True


def metric():
    warmup_runs = 10
    timed_runs = 100

    # Warmup Code
    print('warming up...')
    for _ in range(warmup_runs):
        inputs = generate_input()
        _ = custom_kernel(inputs)
    torch.cuda.synchronize()

    # Timing Code
    total_time = 0.0

    for _ in range(timed_runs):
        inputs = generate_input()

        start_time = time.time()
        custom_output = custom_kernel(inputs)
        torch.cuda.synchronize()
        end_time = time.time()
        total_time += (end_time - start_time)

        ref_output = ref_kernel(inputs)
        torch.cuda.synchronize()
        if not check_implementation(custom_output, ref_output):
            return -1


    custom_duration = total_time / timed_runs

    print(f'Submitted kernel runtime: {custom_duration:.4f} seconds')

    return custom_duration

def main():
    assert (correctness())
    s = metric()

    print(f'score:{s}')

if __name__ == '__main__':
    main()

"""

cu_eval = Path.read_text(Path(__file__).parent / "eval.cu")
