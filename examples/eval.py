import time
import os
import sys
import math

from utils import set_seed
from submission import custom_kernel
from reference import ref_kernel, check_implementation, generate_input

WARMUP_RUNS = 10
TIMED_RUNS = 100


class PopcornOutput:
    def __init__(self, fd: int):
        self.file = os.fdopen(fd, 'w')
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
    
    def print(self, *args, **kwargs):
        print(*args, **kwargs, file=self.file, flush=True)
    
    def log(self, key, value):
        self.print(f"{key}: {value}")


def measure_runtime(logger: PopcornOutput):
    print("warming up...")

    warmup_data = generate_input()
    for _ in range(WARMUP_RUNS):
        custom_kernel(warmup_data)
    
    durations = []

    for _ in range(TIMED_RUNS):
        data = generate_input()
        start = time.time()
        submission_output = custom_kernel(data)
        end = time.time()
        durations.append((end - start) * 1e9)

        reference_output = ref_kernel(data)
        if not check_implementation(submission_output, reference_output):
            logger.log("check", "fail")
            sys.exit(112)
    
    total_duration = sum(durations)
    best = min(durations)
    worst = max(durations)
    average_duration = total_duration / TIMED_RUNS

    variance = sum([(d - average_duration) ** 2 for d in durations])
    standard_deviation = math.sqrt(variance / (TIMED_RUNS - 1))
    standard_error = standard_deviation / math.sqrt(TIMED_RUNS)

    logger.log("check", "pass")
    logger.log("duration.mean", average_duration)
    logger.log("duration.std", standard_deviation)
    logger.log("duration.err", standard_error)
    logger.log("duration.best", best)
    logger.log("duration.worst", worst)

    print(f"average kernel runtime: {average_duration / 1e6} ± {standard_error / 1e6} µs")


def main():
    fd = os.getenv("POPCORN_FD")
    if not fd:
        return 111

    with PopcornOutput(int(fd)) as logger:
        seed = os.getenv("POPCORN_SEED")
        seed = int(seed) if seed else 42

        set_seed(seed)
        data = generate_input()
        reference_output = ref_kernel(data)
        submission_output = custom_kernel(data)

        if not check_implementation(submission_output, reference_output):
            logger.log("check", "fail")
            return 112

        measure_runtime(logger)
        return 0


if __name__ == "__main__":
    sys.exit(main())
