import dataclasses
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from torch.cuda.nvtx import range as nvtx_range

from reference import check_implementation, generate_input
from utils import clear_l2_cache, set_seed


MAX_ITERATIONS_PER_BENCHMARK = 50
BENCHMARK_INPUT_BYTES_TARGET = 256 * 1024 * 1024


class PopcornOutput:
    def __init__(self, fd: int):
        self.file = os.fdopen(fd, "w")
        os.set_inheritable(fd, False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def log(self, key, value):
        print(f"{key}: {value}", file=self.file, flush=True)


@dataclasses.dataclass
class TestCase:
    args: dict
    spec: str


@dataclasses.dataclass
class Stats:
    runs: int
    mean: float
    std: float
    err: float
    best: float
    worst: float


def _combine(a: int, b: int) -> int:
    return int(a + (a + b) * (a + b + 1) // 2)


def get_test_cases(file_name: str, seed: Optional[int]) -> list[TestCase]:
    try:
        content = Path(file_name).read_text()
    except Exception as exc:
        print(f"Could not open test file `{file_name}`: {exc}", file=sys.stderr)
        raise SystemExit(113) from exc

    tests = []
    pattern = r"\s*([a-zA-Z_][a-zA-Z0-9_]*):\s*([a-zA-Z_][a-zA-Z0-9_]*|[+-]?[0-9]+)\s*"
    for line in content.splitlines():
        case = {}
        for part in line.split(";"):
            matched = re.fullmatch(pattern, part)
            if matched is None:
                print(f"invalid test case: '{line}': '{part}'", file=sys.stderr)
                raise SystemExit(113)
            key, value = matched[1], matched[2]
            try:
                value = int(value)
            except ValueError:
                pass
            case[key] = value
        tests.append(TestCase(spec=line, args=case))

    if seed is not None:
        for test in tests:
            if "seed" in test.args:
                test.args["seed"] = _combine(test.args["seed"], seed)
    return tests


def calculate_stats(durations: list[float]) -> Stats:
    runs = len(durations)
    mean = sum(durations) / runs
    variance = sum((duration - mean) ** 2 for duration in durations)
    std = math.sqrt(variance / (runs - 1)) if runs > 1 else 0.0
    return Stats(
        runs=runs,
        mean=mean,
        std=std,
        err=std / math.sqrt(runs),
        best=float(min(durations)),
        worst=float(max(durations)),
    )


def _clone_data(data):
    return data.clone() if isinstance(data, torch.Tensor) else data


def _run_single_test(test: TestCase) -> tuple[bool, str]:
    from submission import custom_kernel

    data = generate_input(**test.args)
    torch.cuda.synchronize()
    output = custom_kernel(_clone_data(data))
    torch.cuda.synchronize()
    return check_implementation(data, output)


def run_testing(logger: PopcornOutput, tests: list[TestCase]) -> int:
    passed = True
    logger.log("test-count", len(tests))
    for index, test in enumerate(tests):
        logger.log(f"test.{index}.spec", test.spec)
        good, message = _run_single_test(test)
        logger.log(f"test.{index}.status", "pass" if good else "fail")
        if message:
            logger.log(
                f"test.{index}.message" if good else f"test.{index}.error", message
            )
        passed = passed and good
    logger.log("check", "pass" if passed else "fail")
    return 0 if passed else 112


def _benchmark_batch_count(test: TestCase) -> int:
    batch = int(test.args.get("batch", 1))
    n = int(test.args.get("n", 1))
    bytes_per_input = batch * n * n * 4
    return max(
        1,
        min(
            MAX_ITERATIONS_PER_BENCHMARK,
            BENCHMARK_INPUT_BYTES_TARGET // bytes_per_input,
        ),
    )


def _make_data_batch(test: TestCase, count: int) -> list[torch.Tensor]:
    args = dict(test.args)
    data_list = []
    for _ in range(count):
        data_list.append(generate_input(**args))
        if "seed" in args:
            args["seed"] += 42
    return data_list


def _run_single_benchmark(
    test: TestCase,
    recheck: bool,
    max_repeats: int,
    max_time_ns: float,
) -> Stats | str:
    from submission import custom_kernel

    data_list = _make_data_batch(test, _benchmark_batch_count(test))
    check_copy = [_clone_data(data) for data in data_list]
    outputs = [custom_kernel(_clone_data(data)) for data in data_list]
    torch.cuda.synchronize()
    for reference_data, output in zip(check_copy, outputs, strict=True):
        good, message = check_implementation(reference_data, output)
        if not good:
            return message

    durations = []
    benchmark_start = time.perf_counter_ns()
    for iteration in range(max_repeats):
        torch.cuda.synchronize()
        clear_l2_cache()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        outputs = [custom_kernel(data) for data in data_list]
        end.record()
        torch.cuda.synchronize()
        durations.append(start.elapsed_time(end) * 1e6 / len(data_list))

        if recheck:
            for reference_data, output in zip(check_copy, outputs, strict=True):
                good, message = check_implementation(reference_data, output)
                if not good:
                    return message

        elapsed = time.perf_counter_ns() - benchmark_start
        if iteration > 1 and elapsed > 1e8:
            stats = calculate_stats(durations)
            if (
                stats.err / stats.mean < 0.001
                or stats.mean * stats.runs > max_time_ns
                or elapsed > 120e9
            ):
                break
    return calculate_stats(durations)


def _log_benchmark(
    logger: PopcornOutput, index: int, test: TestCase, result: Stats | str
) -> bool:
    logger.log(f"benchmark.{index}.spec", test.spec)
    if not isinstance(result, Stats):
        logger.log(f"benchmark.{index}.status", "fail")
        logger.log(f"benchmark.{index}.error", result)
        return False
    for field in dataclasses.fields(Stats):
        logger.log(f"benchmark.{index}.{field.name}", getattr(result, field.name))
    return True


def run_benchmarking(
    logger: PopcornOutput, tests: list[TestCase], leaderboard: bool = False
) -> int:
    _run_single_benchmark(tests[0], False, 20, 1e8)
    max_repeats = 1000 if leaderboard else 200
    max_time_ns = 30e9 if leaderboard else 10e9
    passed = True
    logger.log("benchmark-count", len(tests))
    for index, test in enumerate(tests):
        result = _run_single_benchmark(test, True, max_repeats, max_time_ns)
        passed = _log_benchmark(logger, index, test, result) and passed
        if leaderboard and not passed:
            break
    logger.log("check", "pass" if passed else "fail")
    return 0 if passed else 112


def run_profiling(logger: PopcornOutput, tests: list[TestCase]) -> int:
    from submission import custom_kernel

    test = tests[0]
    logger.log("benchmark-count", 1)
    logger.log("benchmark.0.spec", test.spec)
    with nvtx_range("generate input"):
        data = generate_input(**test.args)
        torch.cuda.synchronize()
    with nvtx_range("custom_kernel"):
        output = custom_kernel(_clone_data(data))
        torch.cuda.synchronize()
    good, message = check_implementation(data, output)
    if not good:
        logger.log("benchmark.0.status", "fail")
        logger.log("benchmark.0.error", message)
    logger.log("check", "pass" if good else "fail")
    return 0 if good else 112


def main() -> int:
    fd = os.getenv("POPCORN_FD")
    if not fd:
        return 111
    if len(sys.argv) < 3:
        return 2

    mode = sys.argv[1]
    seed = os.getenv("POPCORN_SEED")
    os.unsetenv("POPCORN_SEED")
    seed_value = int(seed) if seed else None
    set_seed(seed_value or 42)
    tests = get_test_cases(sys.argv[2], seed_value)

    with PopcornOutput(int(fd)) as logger:
        if mode == "test":
            return run_testing(logger, tests)
        if mode == "benchmark":
            return run_benchmarking(logger, tests)
        if mode == "leaderboard":
            return run_benchmarking(logger, tests, leaderboard=True)
        if mode == "profile":
            return run_profiling(logger, tests)
        return 2


if __name__ == "__main__":
    sys.exit(main())
