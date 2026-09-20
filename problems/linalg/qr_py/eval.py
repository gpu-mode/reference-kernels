import dataclasses
import importlib.util
import math
import multiprocessing
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
from torch.cuda.nvtx import range as nvtx_range

from reference import check_implementation, generate_input
from utils import clear_l2_cache, set_seed

try:
    from task import TestSpec
except ImportError:
    TestSpec = dict


MAX_ITERATIONS_PER_BENCHMARK = 50
BENCHMARK_INPUT_BYTES_TARGET = 256 * 1024 * 1024

# Seed-shift layout for the fresh inputs generated on every timed repeat: each
# (benchmark invocation, repeat, item) triple maps to a unique shift >= 1, so
# no timed input can repeat content from a warmup batch, an earlier repeat, or
# an earlier invocation. Exact for local task seeds; with a server-combined
# base seed the mod below can wrap, making collisions negligible-probability
# rather than impossible. _MAX_TIMED_REPEATS must exceed every max_repeats
# used below; _MAX_BATCH_ITEMS must exceed every _benchmark_batch_count value.
_MAX_TIMED_REPEATS = 1024
_MAX_BATCH_ITEMS = MAX_ITERATIONS_PER_BENCHMARK + 1
# Keep nested _combine results (quadratic in their inputs) inside int64 so
# torch.Generator.manual_seed accepts them.
_SEED_BOUND = 2**63
# Minimum timed samples before the err/mean stability break may fire: the
# untimed per-repeat work (regeneration, recheck) arms the break almost
# immediately on large shapes, where a 3-sample error estimate is noise.
_MIN_STABLE_REPEATS = 10


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


def _timed_repeat_batch(test: TestCase, count: int, seed_salt: int, repeat_index: int):
    # Fresh input content for one timed repeat. The submission process survives
    # across invocations, so a kernel could cache outputs from untimed calls
    # (warmup, earlier repeats) and replay them inside the timed window; the
    # checker would accept the replay because it is correct for that same
    # input. Never reusing content in the timed loop denies every id()- or
    # content-keyed replay: a stale output fails the recheck against the
    # actual, fresh input.
    args = dict(test.args)
    data_list = []
    for item in range(count):
        if "seed" in args:
            shift = 1 + (seed_salt * _MAX_TIMED_REPEATS + repeat_index) * _MAX_BATCH_ITEMS + item
            args["seed"] = _combine(int(test.args["seed"]), shift) % _SEED_BOUND
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
    seed_salt: int,
) -> Stats | Any:
    from submission import custom_kernel

    assert max_repeats < _MAX_TIMED_REPEATS
    count = _benchmark_batch_count(test)
    data_list = _make_data_batch(test, count)
    check_copy = _clone_data(data_list)

    outputs = [custom_kernel(_clone_data(data)) for data in data_list]
    for reference_data, output in zip(check_copy, outputs):
        good, message = check_implementation(reference_data, output)
        if not good:
            return message

    durations = []
    bm_start_time = time.perf_counter_ns()
    for i in range(max_repeats):
        # Regenerate inputs for every timed repeat (see _timed_repeat_batch);
        # generation and cloning stay outside the timed window.
        data_list = _timed_repeat_batch(test, count, seed_salt, i)
        check_copy = _clone_data(data_list) if recheck else None
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
                (stats.runs >= _MIN_STABLE_REPEATS and stats.err / stats.mean < 0.001)
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
    seed_salt: int,
):
    return pool.apply(_run_single_benchmark, (test, recheck, max_repeats, max_time_ns, seed_salt))


def run_benchmarking(logger: PopcornOutput, pool: multiprocessing.Pool, tests: list[TestCase]):
    # Every invocation gets a distinct seed_salt, so content the kernel saw
    # while untimed (the warmup invocation) never reappears in a timed window.
    run_single_benchmark(pool, tests[0], False, 200, 10e7, 0)

    passed = True
    logger.log("benchmark-count", len(tests))
    for idx, test in enumerate(tests):
        logger.log(f"benchmark.{idx}.spec", test.spec)
        # recheck=True: re-validate every timed iteration against that
        # iteration's freshly generated input, not just the pre-timing warmup.
        # Combined with the per-repeat regeneration, a kernel that replays a
        # stored output inside the timed region fails here: the timed content
        # is never something it has seen before. `leaderboard` mode already
        # rechecks; this keeps `benchmark` mode in line.
        result = run_single_benchmark(pool, test, True, 200, 10e9, 1 + idx)
        if isinstance(result, Stats):
            for field in dataclasses.fields(Stats):
                logger.log(f"benchmark.{idx}.{field.name}", getattr(result, field.name))
        else:
            logger.log(f"benchmark.{idx}.status", "fail")
            logger.log(f"benchmark.{idx}.error", result)
            passed = False
    logger.log("check", "pass" if passed else "fail")
    return 0 if passed else 112


def _run_single_profile(test: TestCase):
    from submission import custom_kernel

    with nvtx_range("generate input"):
        data = generate_input(**test.args)
        torch.cuda.synchronize()

    cloned = _clone_data(data)
    with nvtx_range("custom_kernel"):
        output = custom_kernel(cloned)
        torch.cuda.synchronize()

    return check_implementation(data, output)


def run_single_profile(pool: multiprocessing.Pool, test: TestCase):
    return pool.apply(_run_single_profile, (test,))


def run_profiling(logger: PopcornOutput, pool: multiprocessing.Pool, tests: list[TestCase]):
    logger.log("benchmark-count", len(tests))
    test = tests[0]
    logger.log("benchmark.0.spec", test.spec)
    good, message = run_single_profile(pool, test)
    if not good:
        logger.log("benchmark.0.status", "fail")
        logger.log("benchmark.0.error", message)
        logger.log("check", "fail")
        return 112
    logger.log("check", "pass")
    return 0


def _stream_rule_error() -> str | None:
    # KernelBot rejects any submission whose source contains the substring
    # "stream" (case-insensitive) before it ever runs, because non-default
    # streams can escape the timed region. That gate lives in the submission
    # intake, so locally a stream-using kernel passes every mode and is only
    # rejected on its first remote submission. Mirror the rule here for parity.
    spec = importlib.util.find_spec("submission")
    if spec is not None and spec.origin:
        submission_path = Path(spec.origin)
    else:
        submission_path = Path(__file__).resolve().with_name("submission.py")
    try:
        source = submission_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if "stream" in source.lower():
        return (
            "submission.py contains the substring 'stream' (case-insensitive), "
            "which the leaderboard rejects at submission time in any form, "
            "including comments and identifiers. Work on non-default CUDA "
            "streams is not allowed; remove every occurrence before submitting."
        )
    return None


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
        stream_error = _stream_rule_error()
        if stream_error is not None:
            section = "test" if mode == "test" else "benchmark"
            logger.log(f"{section}-count", 1)
            logger.log(f"{section}.0.spec", "stream-rule")
            logger.log(f"{section}.0.status", "fail")
            logger.log(f"{section}.0.error", stream_error)
            logger.log("check", "fail")
            print(stream_error, file=sys.stderr)
            return 112

        mp_context = multiprocessing.get_context("spawn")
        with mp_context.Pool(1) as pool:
            if mode == "test":
                return run_testing(logger, pool, tests)
            if mode == "benchmark":
                return run_benchmarking(logger, pool, tests)
            if mode == "leaderboard":
                # Warmup salts 1..len(tests); timed salts start after them, so
                # no timed repeat reuses content from any warmup invocation.
                for idx, test in enumerate(tests):
                    run_single_benchmark(pool, test, False, 1000, 5e8, 1 + idx)
                logger.log("benchmark-count", len(tests))
                passed = True
                for idx, test in enumerate(tests):
                    logger.log(f"benchmark.{idx}.spec", test.spec)
                    result = run_single_benchmark(pool, test, True, 1000, 30e9, 1 + len(tests) + idx)
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
                return run_profiling(logger, pool, tests)
            return 2


if __name__ == "__main__":
    sys.exit(main())
