########
# Evaluation scripts to run for leaderboard results
########

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
    inputs = generate_input()
    start_time = time.time()
    for _ in range(timed_runs):
        _ = custom_kernel(inputs)
    torch.cuda.synchronize()
    end_time = time.time()

    custom_duration = (end_time - start_time) / timed_runs

    print(f'Submitted kernel runtime: {custom_duration:.4f} seconds')

    return custom_duration

def main():
    assert (correctness())
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
        custom_kernel(data);
    }
    cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < TIMED_RUNS; i++) {
        auto data = generate_input();
        custom_kernel(data);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    using double_duration = std::chrono::duration<double>;
    auto duration = std::chrono::duration_cast<double_duration>(end - start).count() / TIMED_RUNS;
    std::cout << "submitted kernel runtime: " << duration << " seconds" << std::endl;
    return duration;
}

int main() {
    auto data = generate_input();
    auto reference_output = ref_kernel(data);
    auto submission_output = custom_kernel(data);

    if (!check_implementation(submission_output, reference_output)) {
        std::cout << "check_implementation failed" << std::endl;
        return 1;
    }

    float s = measure_runtime();
    std::cout << "score: " << s << std::endl;

    return 0;
}

"""
