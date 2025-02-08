#include <chrono>
#include <iostream>
#include <cstdint>
#include <vector>
#include <numeric>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <regex>
#include <cassert>
#include <variant>
#include <tuple>
#include "utils.h"
#include "reference.cuh"

// forward declaration for user submission
output_t custom_kernel(input_t data);

#define WARMUP_RUNS 10
#define TIMED_RUNS 100

namespace
{
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

    void log(const std::string& key, const char* value) {
        printf("%s: %s\n", key.c_str(), value);
    }

    template<class T>
    void log(const std::string& key, T&& value) {
        log(key, std::to_string(value).c_str());
    }

    std::unique_ptr<std::FILE, Closer> File;
};

template<class F>
struct extract_signature_helper;

template<class R, class... Args>
struct extract_signature_helper<R(*)(Args...)> {
    using tuple_t = std::tuple<std::remove_const_t<std::remove_reference_t<Args>>...>;
};

struct TestCase {
    using Parameter = typename extract_signature_helper<decltype(&generate_input)>::tuple_t;
    static_assert(std::tuple_size<Parameter>() == ArgumentNames.size(), "Mismatch in argument name count");
    std::string spec;
    Parameter params;
};

template<class T>
void assign_value(T& target, const std::string& raw) {
    if constexpr (std::is_same_v<T, std::string>) {
        target = raw;
    } else {
        static_assert(std::is_same_v<T, int>, "Test arguments must be integers or strings");
        target = std::stoi(raw);
    }
}

template<std::size_t Index = ArgumentNames.size()>
bool set_param_value(TestCase::Parameter& param, const std::string& key, const std::string& raw, std::integral_constant<std::size_t, Index> = {}) {
    if(key == ArgumentNames[Index-1]) {
        assign_value(std::get<Index-1>(param), raw);
        return true;
    }
    if constexpr (Index != 1) {
        return set_param_value(param, key, raw, std::integral_constant<std::size_t, Index-1>{});
    }
    return false;
}

// reads a line from a std::FILE
std::string read_line(std::FILE* file) {
    std::string buf;
    while(true) {
        int next = std::fgetc(file);
        if (next == '\n' || next == EOF) {
            return buf;
        }
        buf.push_back((char)next);
    }

}

TestCase parse_test_case(const std::string& line) {
    // match a key-value pair of integer or string value
    static std::regex match_entry(R"(\s*([a-zA-Z]+):\s*([a-zA-Z]+|[+-]?[0-9]+)\s*)");

    TestCase tc;
    tc.spec = line;

    // split line into individual arguments
    std::vector<std::string> parts;
    std::size_t pos = 0;
    while(pos != std::string::npos) {
        std::size_t next = line.find(';', pos);
        parts.push_back(line.substr(pos, next));
        pos = next == std::string::npos ? next : next + 1;
    }

    // split arguments into kv pairs
    for(const std::string& arg : parts) {
        std::smatch m;
        if(!std::regex_match(arg, m, match_entry)) {
            std::cerr << "invalid test case: ''" << line << "'': '" << arg << "'" << std::endl;
            std::exit(ExitCodes::EXIT_TEST_SPEC);
        }

        // TODO check that we get all values
        // TODO check that no value is duplicate

        std::string key = std::string(m[1].first, m[1].second);
        std::string value = std::string(m[2].first, m[2].second);
        if(!set_param_value(tc.params, key, value)) {
            std::cerr << "invalid test case: ";
            std::cerr << "argument name '" << key << "' is invalid" << std::endl;
            std::exit(ExitCodes::EXIT_TEST_SPEC);
        }
    }
    return tc;
}

PopcornOutput open_logger() {
    PopcornOutput logger;
    const char *output_fd = std::getenv("POPCORN_FD");
    if (output_fd) {
        int fd = std::stoi(output_fd);
        logger.File.reset(::fdopen(fd, "w"));
    } else {
        std::cerr << "Missing POPCORN_FD file descriptor." << std::endl;
        std::exit(ExitCodes::EXIT_PIPE_FAIL);
    }
    unsetenv("POPCORN_FD");
    return logger;
}

int get_seed() {
    const char *seed_str = std::getenv("POPCORN_SEED");
    int seed = 42;
    if (seed_str) {
        seed = std::stoi(seed_str);
    }
    unsetenv("POPCORN_SEED");
    return seed;
}

std::vector<TestCase> get_test_cases(const std::string& tests_file_name) {
    std::unique_ptr<std::FILE, Closer> test_case_file;
    test_case_file.reset(::fopen(tests_file_name.c_str(), "r"));

    if(!test_case_file) {
        std::error_code ec(errno, std::system_category());
        std::cerr << "Could not open test file`" << tests_file_name << "`: " << ec.message() << std::endl;
        std::exit(ExitCodes::EXIT_PIPE_FAIL);
    }

    std::vector<TestCase> tests;
    while(true) {
        std::string line = read_line(test_case_file.get());
        tests.push_back(parse_test_case(line));

        // have we reached eof
        int peek = std::getc(test_case_file.get());
        if(peek != EOF) {
            std::ungetc(peek, test_case_file.get());
        } else {
            return tests;
        }
    }
}

template<class R, class... Args, std::size_t... Indices>
R call_generate_input_helper(const TestCase& tc, R(*func)(Args...), const std::index_sequence<Indices...>&) {
    return func(std::get<Indices>(tc.params)...);
}

template<class R, class... Args>
R call_generate_input(const TestCase& tc, R(*func)(Args...)) {
    return call_generate_input_helper(tc, func, std::make_index_sequence<sizeof...(Args)>{});
}

void warm_up(const TestCase& test) {
    using std::chrono::milliseconds;
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;

    {
        auto warmup_data = call_generate_input(test, &generate_input);
        // warm up for at least 200 milliseconds
        auto start = high_resolution_clock::now();
        while(duration_cast<milliseconds>(high_resolution_clock::now() - start).count() < 200) {
            // discard result; this is just warmup, we don't care what it returns
            CUDA_CHECK(cudaDeviceSynchronize());
            (void)custom_kernel(warmup_data);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
}

struct BenchmarkStats {
    int runs;
    double mean;
    double std;
    double err;
    double best;
    double worst;
};

BenchmarkStats calculate_stats(const std::vector<std::int64_t>& durations) {
    /**
     * Aggregate runtime statistics for a particular set of runtimes.
     * 
     * @param durations A sequence of runtimes for a particular benchmark.
     * @return A set of statistics for this particular benchmark case.
     */
    int runs = (int)durations.size();
    // calculate duration statistics
    std::int64_t total_duration = std::accumulate(durations.begin(), durations.end(), (std::int64_t)0);
    std::int64_t best = *std::min_element(durations.begin(), durations.end());
    std::int64_t worst = *std::max_element(durations.begin(), durations.end());
    double average_duration = (double)total_duration / runs;

    double variance = 0.0;
    for(auto d : durations) {
        variance += std::pow((double)d - average_duration, 2);
    }

    // sample standard deviation with Bessel's correction
    double standard_deviation = std::sqrt(variance / (runs - 1));
    // standard error of the mean
    double standard_error = standard_deviation / std::sqrt(runs);

    return {runs, average_duration, standard_deviation, standard_error, (double)best, (double)worst};
}

using BenchmarkResults = std::variant<BenchmarkStats, TestReporter>;

BenchmarkResults benchmark(const TestCase& test_case, bool test_correctness, int max_repeats, float max_time_ns) {
    /**
     * For a particular test case, check correctness (if applicable) and grab runtime results.
     * 
     * @param test_case Test case object.
     * @param test_correctness Flag for whether to explicitly check functional correctness
     * @param max_repeats Number of trials to repeat
     * @param max_time_ns Timeout time
     * @return A set of statistics for this particular benchmark case or a TestReporter result.
     */
    std::vector<std::int64_t> durations;
    durations.reserve(max_repeats);

    // generate input data once
    auto data = call_generate_input(test_case, &generate_input);
    auto copy = data;

    // first, one obligatory correctness check
    {
        TestReporter reporter;
        auto submission_output = custom_kernel(std::move(copy));
        check_implementation(reporter, data, submission_output);
        if(!reporter.has_passed()) {
            return reporter;
        }
    }

    // now, do multiple timing runs
    // there is an upper bound of `max_repeats` runs, and a lower bound of 3 runs;
    // otherwise, we repeat until we either measure at least max_time_ns nanoseconds,
    // or the relative error of the mean is below 1%.

    for(int i = 0; i < max_repeats; ++i) {
        // stricter checking for leaderboard submissions
        if(test_correctness) {
            data = call_generate_input(test_case, &generate_input);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        copy = data;
        auto start = std::chrono::high_resolution_clock::now();
        // move data into custom_kernel, so that if custom_kernel takes large std::vectors or similar by value,
        // we're not measuring the copy overhead.
        auto submission_output = custom_kernel(std::move(copy));
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();

        if(test_correctness) {
            TestReporter reporter;
            copy = data;
            check_implementation(reporter, copy, submission_output);

            if(!reporter.has_passed()) {
                return reporter;
            }
        }

        durations.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

        if(i > 1) {
            auto stats = calculate_stats(durations);
            // if we have enough data for an error < 1%
            // or if the total running time exceeds 10 seconds
            if((stats.err / stats.mean < 0.01) || (stats.mean * stats.runs > max_time_ns)) {
                break;
            }
        }
    }

    return calculate_stats(durations);
}

} // namespace


int run_testing(PopcornOutput& logger, const std::vector<TestCase>& tests) {
    /**
     * Executes the actual test case code, and check for correctness.
     * 
     * @param logger A PopcornOutput object used for logging test results.
     * @param tests A vector of TestCase objects representing the test cases to be executed.
     * @return An integer representing the exit status: EXIT_SUCCESS if all tests pass, otherwise EXIT_TEST_FAIL.
     */
    bool pass = true;
    logger.log("test-count", tests.size());
    for (int i = 0; i < tests.size(); ++i) {
        auto& tc = tests.at(i);
        logger.log("test." + std::to_string(i) + ".spec", tc.spec.c_str());
        auto data = call_generate_input(tc, &generate_input);
        auto copy = data;
        CUDA_CHECK(cudaDeviceSynchronize());
        auto submission_output = custom_kernel(std::move(data));
        CUDA_CHECK(cudaDeviceSynchronize());

        TestReporter reporter;
        check_implementation(reporter, copy, submission_output);

        // log test status
        if (!reporter.has_passed()) {
            logger.log("test." + std::to_string(i) + ".status", "fail");
            logger.log("test." + std::to_string(i) + ".error", reporter.message().c_str());
            pass = false;
        } else {
            logger.log("test." + std::to_string(i) + ".status", "pass");
        }
    }

    if(pass) {
        logger.log("check", "pass");
        return EXIT_SUCCESS;
    } else {
        logger.log("check", "fail");
        return ExitCodes::EXIT_TEST_FAIL;
    }
};

int run_benchmarking(PopcornOutput& logger, const std::vector<TestCase>& tests) {
    /**
     * Executes benchmarking code for a CUDA Kernel and logs runtimes.
     * 
     * @param logger A PopcornOutput object used for logging benchmark results.
     * @param tests A vector of TestCase objects representing the test cases to be benchmarked.
     * @return An integer representing the exit status: EXIT_SUCCESS if all benchmarks pass, otherwise EXIT_TEST_FAIL.
     */
    warm_up(tests.front());
    bool pass = true;
    logger.log("benchmark-count", tests.size());
    for (int i = 0; i < tests.size(); ++i) {
        const TestCase& tc = tests.at(i);
        logger.log("benchmark." + std::to_string(i) + ".spec", tc.spec.c_str());

        auto result = benchmark(tc, false, 100, 10e9);
        if(std::holds_alternative<BenchmarkStats>(result)) {
            auto &stats = std::get<BenchmarkStats>(result);
            logger.log("benchmark." + std::to_string(i) + ".status", "pass");
            logger.log("benchmark." + std::to_string(i) + ".runs", stats.runs);
            logger.log("benchmark." + std::to_string(i) + ".mean", stats.mean);
            logger.log("benchmark." + std::to_string(i) + ".std", stats.std);
            logger.log("benchmark." + std::to_string(i) + ".err", stats.err);
            logger.log("benchmark." + std::to_string(i) + ".best", stats.best);
            logger.log("benchmark." + std::to_string(i) + ".worst", stats.worst);
        } else {
            auto& rep = std::get<TestReporter>(result);
            logger.log("benchmark." + std::to_string(i) + ".status", "fail");
            logger.log("benchmark." + std::to_string(i) + ".error", rep.message().c_str());
            pass = false;
        }
    }

    if(pass) {
        logger.log("check", "pass");
        return EXIT_SUCCESS;
    } else {
        logger.log("check", "fail");
        return ExitCodes::EXIT_TEST_FAIL;
    }
}

int main(int argc, const char* argv[]) {
    // setup
    PopcornOutput logger = open_logger();
    int seed = get_seed();

    if(argc < 3) {
        return ExitCodes::USAGE_ERROR;
    }

    std::string mode = argv[1];

    std::vector<TestCase> tests = get_test_cases(argv[2]);

    if(mode == "test") {
        return run_testing(logger, tests);
    }

    if (mode == "benchmark") {
      return run_benchmarking(logger, tests);
    }

    if (mode == "leaderboard" ) {
        warm_up(tests.front());
        auto result = benchmark(tests.back(), true, 100, 30e9);
        if(std::holds_alternative<BenchmarkStats>(result)) {
            logger.log("benchmark-count", 1);
            auto& stats = std::get<BenchmarkStats>(result);
            logger.log("benchmark.0.spec", tests.back().spec.c_str());
            logger.log("benchmark.0.runs", stats.runs);
            logger.log("benchmark.0.mean", stats.mean);
            logger.log("benchmark.0.std", stats.std);
            logger.log("benchmark.0.err", stats.err);
            logger.log("benchmark.0.best", stats.best);
            logger.log("benchmark.0.worst", stats.worst);
            logger.log("check", "pass");
        } else {
            logger.log("test-count", 1);
            auto& rep = std::get<TestReporter>(result);
            logger.log("test.0.status", "fail");
            logger.log("test.0.error", rep.message().c_str());
        }
    } else {
        // TODO implement script and profile modes
        std::cerr << "Unknown evaluation mode '" << mode << "'" << std::endl;
        return ExitCodes::USAGE_ERROR;
    }

    return EXIT_SUCCESS;
}
