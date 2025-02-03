#include <chrono>
#include <iostream>
#include <cstdint>
#include <vector>
#include <numeric>
#include <algorithm>
#include <memory>

#include "utils.h"
#include "reference.cuh"

// forward declaration for user submission
output_t custom_kernel(input_t data);

#define WARMUP_RUNS 10
#define TIMED_RUNS 100

namespace {
    struct Closer {
        void operator()(std::FILE* file) {
            std::fclose(file);
        }
    };

    struct PopcornOutput {
        template<class... Args>
        void printf(Args&&... args) {
            ::fprintf(File.get(), std::forward<Args>(args)...);
        }

        void log(const char* key, const char* value) {
            printf("%s: %s\n", key, value);
        }

        template<class T>
        void log(const char* key, T&& value) {
            log(key, std::to_string(value).c_str());
        }

        std::unique_ptr<std::FILE, Closer> File;
    };
}

static void measure_runtime(PopcornOutput& logger, std::mt19937& rng) {
    std::cout << "warming up..." << std::endl;

    {
        auto warmup_data = generate_input(rng());
        for (int i = 0; i < WARMUP_RUNS; i++) {
            // discard result; this is just warmup, we don't care what it returns
            (void)custom_kernel(warmup_data);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    std::vector<std::int64_t> durations;
    durations.reserve(TIMED_RUNS);

    for (int i = 0; i < TIMED_RUNS; i++) {
        auto data = generate_input(rng());

        // make a copy of the input data to be used by the reference implementation
        auto copy = data;

        auto start = std::chrono::high_resolution_clock::now();
        // move data into custom_kernel, so that if custom_kernel takes large std::vectors or similar by value,
        // we're not measuring the copy overhead.
        auto submission_output = custom_kernel(std::move(data));
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();

        durations.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

        auto reference_output = ref_kernel(copy);
        if (!check_implementation(submission_output, reference_output)) {
            logger.log("check", "fail");
            std::exit(112);
        }

    }

    // calculate duration statistics
    std::int64_t total_duration = std::accumulate(durations.begin(), durations.end(), (std::int64_t)0);
    std::int64_t best = *std::min_element(durations.begin(), durations.end());
    std::int64_t worst = *std::max_element(durations.begin(), durations.end());
    double average_duration = (double)total_duration / TIMED_RUNS;

    double variance = 0.0;
    for(auto d : durations) {
        variance += std::pow((double)d - average_duration, 2);
    }

    // sample standard deviation with Bessel's correction
    double standard_deviation = std::sqrt(variance / (TIMED_RUNS - 1));
    // standard error of the mean
    double standard_error = standard_deviation / std::sqrt(TIMED_RUNS);

    logger.log("check", "pass");
    logger.log("duration.mean", average_duration);
    logger.log("duration.std", standard_deviation);
    logger.log("duration.err", standard_error);
    logger.log("duration.best", best);
    logger.log("duration.worst", worst);


    std::cout << "average kernel runtime: " << average_duration / 1e6 << " ± " << standard_error / 1e6 << " µs" << std::endl;
}

int main() {
    const char *output_fd = std::getenv("POPCORN_FD");
    PopcornOutput logger;
    if (output_fd) {
        int fd = std::stoi(output_fd);
        logger.File.reset(::fdopen(fd, "w"));
    } else {
        return 111;
    }

    // get the seed
    const char *seed_str = std::getenv("POPCORN_SEED");
    int seed = 42;
    if (seed_str) {
        seed = std::stoi(output_fd);
    }

    std::mt19937 rng(seed);
    auto data = generate_input(rng());
    auto reference_output = ref_kernel(data);
    auto submission_output = custom_kernel(data);

    if (!check_implementation(submission_output, reference_output)) {
        logger.log("check", "fail");
        return 112;
    }

    measure_runtime(logger, rng);
    return 0;
}
