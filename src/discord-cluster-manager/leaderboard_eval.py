########
# Evaluation scripts to run for leaderboard results
########

py_eval = """
import torch
import time
from reference import ref_kernel, generate_input
from train import custom_kernel


def check_implementation() -> bool:
    for _ in range(10):  # check multiple times
        input_tensors = generate_input()
        for input in input_tensors:
            custom_output = custom_kernel(input)
            ref_output = ref_kernel(input)

            if not torch.allclose(custom_output, ref_output, atol=1e-5):
                print('mismatch found! custom implementation doesnt match reference.')
                return False

    print('custom implementation matches the reference implementation.')
    return True


def metric():
    warmup_runs = 10
    timed_runs = 100

    # warmup
    print('warming up...')
    for _ in range(warmup_runs):
        input_tensors = generate_input()
        for input in input_tensors:
            _ = custom_kernel(input)
            _ = ref_kernel(input)

    # timing
    print('timing custom implementation...')
    input_tensor = generate_input()
    start_time = time.time()
    for _ in range(timed_runs):
        for input in input_tensors:
            _ = custom_kernel(input)

    custom_duration = (time.time() - start_time) / timed_runs

    print(f'submitted kernel runtime: {custom_duration:.4f} seconds')

    return custom_duration

def main():
    assert (check_implementation())
    s = metric()

    print(f'score:{s}')

if __name__ == '__main__':
    main()

"""

cu_eval = """
#include <chrono>
#include <iostream>

#include "reference.cuh"
#include "train.cuh"

#define WARMUP_RUNS 10
#define TIMED_RUNS 100


float measure_runtime() {
    std::cout << "warming up..." << std::endl;

    for (int i = 0; i < WARMUP_RUNS; i++) {
        auto data = generate_input();
        submission(data);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < TIMED_RUNS; i++) {
        auto data = generate_input();
        submission(data);
    }

    auto end = std::chrono::high_resolution_clock::now();

    using double_duration = std::chrono::duration<double>;
    auto duration = std::chrono::duration_cast<double_duration>(end - start).count() / TIMED_RUNS;
    std::cout << "submitted kernel runtime: " << duration << " seconds" << std::endl;
    return duration;
}

int main() {
    auto data = generate_input();
    auto reference_output = reference(data);
    auto submission_output = submission(data);

    if (!check_implementation(submission_output, reference_output)) {
        std::cout << "check_implementation failed" << std::endl;
        return 1;
    }

    float s = measure_runtime();
    std::cout << "score: " << s << std::endl;

    return 0;
}

"""
