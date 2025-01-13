import math
import os
import sys
import time

import torch
from reference import check_implementation, generate_input, ref_kernel
from train import custom_kernel


class PopcornLogger:
    def __init__(self, fd):
        self.channel = open(fd, "w")

    def log(self, key: str, value):
        print(f"{key}: {value}\n", file=self.channel)


def correctness() -> bool:
    for _ in range(10):  # check multiple times
        inputs = generate_input()

        custom_output = custom_kernel(inputs)
        ref_output = ref_kernel(inputs)

        if not check_implementation(custom_output, ref_output):
            return False

    print("custom implementation matches the reference implementation.")
    return True


def metric(logger: PopcornLogger):
    warmup_runs = 10
    timed_runs = 100

    # Warmup Code
    print("warming up...")
    for _ in range(warmup_runs):
        inputs = generate_input()
        _ = custom_kernel(inputs)
    torch.cuda.synchronize()

    # Timing Code
    times = []

    for _ in range(timed_runs):
        inputs = generate_input()

        start_time = time.time()
        custom_output = custom_kernel(inputs)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)

        ref_output = ref_kernel(inputs)
        torch.cuda.synchronize()
        if not check_implementation(custom_output, ref_output):
            logger.log("check", "fail")
            exit(112)

    total_time = sum(times)
    average_duration = total_time / timed_runs
    variance = sum(map(lambda x: (x - average_duration) ** 2, times))  # noqa
    standard_deviation = math.sqrt(variance / (timed_runs - 1))
    standard_error = standard_deviation / math.sqrt(timed_runs)

    logger.log("check", "pass")
    logger.log("duration.mean", average_duration * 1e9)
    logger.log("duration.std", standard_deviation * 1e9)
    logger.log("duration.err", standard_error * 1e9)
    logger.log("duration.best", min(times) * 1e9)
    logger.log("duration.worst", max(times) * 1e9)

    print(f"Submitted kernel runtime: {average_duration:.4f} ± {standard_error:.4} seconds")


def main():
    try:
        logger = PopcornLogger(int(os.environ["POPCORN_FD"]))
    except Exception as e:
        print(e, file=sys.stderr)
        exit(111)

    if not correctness():
        logger.log("check", "fail")
        exit(112)
    metric(logger)


if __name__ == "__main__":
    main()
