import dataclasses
import math
import multiprocessing
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch

from reference import check_implementation, generate_input
from utils import clear_l2_cache, set_seed

try:
    from task import TestSpec
except ImportError:
    TestSpec = dict


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

    def print(self, *args, **kwargs):
        print(*args, **kwargs, file=self.file, flush=True)

    def log(self, key, value):
        self.print(f"{key}: {value}")


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
        exit(113)

    tests = []
    match = r"\s*([a-zA-Z]+):\s*([a-zA-Z]+|[+-]?[0-9]+)\s*"
    for line in content.splitlines():
        case = {}
        for part in line.split(";"):
            matched = re.match(match, part)
            if not re.fullmatch(match, part):
                print(f"invalid test case: '{line}': '{part}'", file=sys.stderr)
                exit(113)
            key = matched[1]
            val = matched[2]
            try:
                val = int(val)
            except ValueError:
                pass
            case[key] = val
        tests.append(TestCase(spec=line, args=case))

    if seed is not None:
        for test in tests:
            if "seed" in test.args:
                test.args["seed"] = _combine(test.args["seed"], seed)
    return tests


def calculate_stats(durations: list[float]) -> Stats:
    runs = len(durations)
    total = sum(durations)
    avg = total / runs
    variance = sum((x - avg) ** 2 for x in durations)
    std = math.sqrt(variance / (runs - 1)) if runs > 1 else 0.0
    err = std / math.sqrt(runs) if runs > 0 else 0.0
    return Stats(
        runs=runs,
        mean=avg,
        std=std,
        err=err,
        best=float(min(durations)),
        worst=float(max(durations)),
    )


def _clone_data(data):
    if isinstance(data, tuple):
        return tuple(_clone_data(x) for x in data)
    if isinstance(data, list):
        return [_clone_data(x) for x in data]
    if isinstance(data, dict):
        return {k: _clone_data(v) for k, v in data.items()}
    if isinstance(data, torch.Tensor):
        return data.clone()
    return data


def _run_single_test(test: TestCase):
    from submission import custom_kernel

    data = generate_input(**test.args)
    torch.cuda.synchronize()
    output = custom_kernel(_clone_data(data))
    torch.cuda.synchronize()
    return check_implementation(data, output)


def run_single_test(pool: multiprocessing.Pool, test: TestCase):
    return pool.apply(_run_single_test, (test,))


def run_testing(logger: PopcornOutput, pool: multiprocessing.Pool, tests: list[TestCase]):
    passed = True
    logger.log("test-count", len(tests))
    for idx, test in enumerate(tests):
        logger.log(f"test.{idx}.spec", test.spec)
        good, message = run_single_test(pool, test)
        if good:
            logger.log(f"test.{idx}.status", "pass")
            if message:
                logger.log(f"test.{idx}.message", message)
        else:
            logger.log(f"test.{idx}.status", "fail")
            logger.log(f"test.{idx}.error", message)
            passed = False
    logger.log("check", "pass" if passed else "fail")
    return 0 if passed else 112


def _make_data_batch(test: TestCase, count: int):
    args = dict(test.args)
    data_list = []
    for _ in range(count):
        if "seed" in args:
            args["seed"] += 42
        data_list.append(generate_input(**args))
    return data_list


def _benchmark_batch_count(test: TestCase) -> int:
    batch = int(test.args.get("batch", 1))
    n = int(test.args.get("n", 1))
    # Input storage is A. Keep the generated batch modest
    # because large QR cases are already batched inside a single input.
    bytes_per_input = (batch * n * n) * 4
    if bytes_per_input <= 0:
        return 1
    return max(1, min(MAX_ITERATIONS_PER_BENCHMARK, BENCHMARK_INPUT_BYTES_TARGET // bytes_per_input))


def _run_single_benchmark(
    test: TestCase,
    recheck: bool,
    max_repeats: int,
    max_time_ns: float,
) -> Stats | Any:
    from submission import custom_kernel

    data_list = _make_data_batch(test, _benchmark_batch_count(test))
    check_copy = _clone_data(data_list)

    outputs = [custom_kernel(_clone_data(data)) for data in data_list]
    for reference_data, output in zip(check_copy, outputs):
        good, message = check_implementation(reference_data, output)
        if not good:
            return message

    durations = []
    bm_start_time = time.perf_counter_ns()
    for i in range(max_repeats):
        torch.cuda.synchronize()
        clear_l2_cache()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        outputs = [custom_kernel(data) for data in data_list]
        end_event.record()
        torch.cuda.synchronize()
        durations.append(start_event.elapsed_time(end_event) * 1e6 / len(data_list))

        if recheck:
            for reference_data, output in zip(check_copy, outputs):
                good, message = check_implementation(reference_data, output)
                if not good:
                    return message

        total_bm_duration = time.perf_counter_ns() - bm_start_time
        if i > 1 and total_bm_duration > 1e8:
            stats = calculate_stats(durations)
            if (
                stats.err / stats.mean < 0.001
                or stats.mean * stats.runs > max_time_ns
                or total_bm_duration > 120e9
            ):
                break

    return calculate_stats(durations)


def run_single_benchmark(
    pool: multiprocessing.Pool,
    test: TestCase,
    recheck: bool,
    max_repeats: int,
    max_time_ns: float,
):
    return pool.apply(_run_single_benchmark, (test, recheck, max_repeats, max_time_ns))


def run_benchmarking(logger: PopcornOutput, pool: multiprocessing.Pool, tests: list[TestCase]):
    run_single_benchmark(pool, tests[0], False, 200, 10e7)

    passed = True
    logger.log("benchmark-count", len(tests))
    for idx, test in enumerate(tests):
        logger.log(f"benchmark.{idx}.spec", test.spec)
        # recheck=True: re-validate the output of every timed iteration, not just
        # the pre-timing warmup. Without this, the timed loop (which for the
        # low-`count` shapes reuses one input object across all repeats) never
        # re-checks its outputs, so a kernel that diverges only inside the timed
        # region -- e.g. one that caches and replays an output keyed on the
        # reused input -- is scored as fast without ever being caught locally.
        # `leaderboard` mode already rechecks; this brings `benchmark` mode in
        # line so a wrong timed output fails here too.
        result = run_single_benchmark(pool, test, True, 200, 10e9)
        if isinstance(result, Stats):
            for field in dataclasses.fields(Stats):
                logger.log(f"benchmark.{idx}.{field.name}", getattr(result, field.name))
        else:
            logger.log(f"benchmark.{idx}.status", "fail")
            logger.log(f"benchmark.{idx}.error", result)
            passed = False
    logger.log("check", "pass" if passed else "fail")
    return 0 if passed else 112


def main():
    fd = os.getenv("POPCORN_FD")
    if not fd:
        return 111
    if len(sys.argv) < 3:
        return 2

    mode = sys.argv[1]
    seed = os.getenv("POPCORN_SEED")
    os.unsetenv("POPCORN_SEED")
    seed = int(seed) if seed else None
    set_seed(seed or 42)
    tests = get_test_cases(sys.argv[2], seed)

    with PopcornOutput(int(fd)) as logger:
        mp_context = multiprocessing.get_context("spawn")
        with mp_context.Pool(1) as pool:
            if mode == "test":
                return run_testing(logger, pool, tests)
            if mode == "benchmark":
                return run_benchmarking(logger, pool, tests)
            if mode == "leaderboard":
                for test in tests:
                    run_single_benchmark(pool, test, False, 1000, 5e8)
                logger.log("benchmark-count", len(tests))
                passed = True
                for idx, test in enumerate(tests):
                    logger.log(f"benchmark.{idx}.spec", test.spec)
                    result = run_single_benchmark(pool, test, True, 1000, 30e9)
                    if isinstance(result, Stats):
                        for field in dataclasses.fields(Stats):
                            logger.log(f"benchmark.{idx}.{field.name}", getattr(result, field.name))
                    else:
                        logger.log(f"benchmark.{idx}.status", "fail")
                        logger.log(f"benchmark.{idx}.error", str(result))
                        passed = False
                        break
                logger.log("check", "pass" if passed else "fail")
                return 0 if passed else 112
            if mode == "profile":
                logger.log("check", "fail")
                logger.log("error", "profile mode is not implemented for qr eval.py")
                return 2
            return 2


if __name__ == "__main__":
    sys.exit(main())
