#include <chrono>
#include <iostream>
#include <cstdint>
#include <vector>
#include <numeric>

#include "reference.cuh"
#include "train.cuh"

#define WARMUP_RUNS 10
#define TIMED_RUNS 100

// checks that a CUDA API call returned successfully, otherwise prints an error message and exits.
static void cuda_check(cudaError_t status, const char* expr, const char* file, int line, const char* function)
{
    if(status != cudaSuccess) {
        std::cerr << "CUDA error (" << (int)status << ") while evaluating expression "
                  << expr << " at "
                  << file << '('
                  << line << ") in `"
                  << function << "`: "
                  << cudaGetErrorString(status) << std::endl;
        // following pytest convention, exit code 3 means internal error
        std::exit(3);
    }
}

#define cuda_check(expr) cuda_check(expr, #expr, __FILE__, __LINE__, __FUNCTION__)

double measure_runtime() {
    std::cout << "warming up..." << std::endl;

    for (int i = 0; i < WARMUP_RUNS; i++) {
        auto data = generate_input();
        // discard result; this is just warmup, we don't care what it returns
        (void)custom_kernel(data);
    }
    cuda_check(cudaDeviceSynchronize());

    std::vector<std::int64_t> durations;
    durations.reserve(TIMED_RUNS);

    for (int i = 0; i < TIMED_RUNS; i++) {
        auto data = generate_input();
        // make a copy of the input data to be used by the reference implementation
        auto copy = data;

        auto start = std::chrono::high_resolution_clock::now();
        // move data into custom_kernel, so that if custom_kernel takes large std::vectors or similar by value,
        // we're not measuring the copy overhead.
        auto submission_output = custom_kernel(std::move(data));
        cuda_check(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();

        durations.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

        auto reference_output = ref_kernel(copy);
        if (!check_implementation(submission_output, reference_output)) {
            std::cout << "check_implementation failed" << std::endl;
            // following pytest convention, code 1 means that tests failed
            std::exit(1);
        }

    }

    // calculate duration statistics
    std::int64_t total_duration = std::accumulate(durations.begin(), durations.end(), (std::int64_t)0);

    double average_duration = (double)total_duration / 1e9 / TIMED_RUNS;
    std::cout << "submitted kernel runtime: " << average_duration << " seconds" << std::endl;
    return average_duration;
}

int main() {
    auto data = generate_input();
    auto reference_output = ref_kernel(data);
    auto submission_output = custom_kernel(data);

    if (!check_implementation(submission_output, reference_output)) {
        std::cout << "check_implementation failed" << std::endl;
        return 1;
    }

    double s = measure_runtime();
    if (s < 0) {
        return 1;
    }

    std::cout << "score: " << s << std::endl;

    return 0;
}
