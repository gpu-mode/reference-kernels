import base64
import dataclasses
import multiprocessing
import random
import re
import time
import os
import sys
import math
import random

# Disable CuTe DSL file caching for more stable benchmarking
os.environ["CUTE_DSL_DISABLE_FILE_CACHING"] = "1"


def _init_worker():
    """Initialize worker process with correct env vars."""
    os.environ["CUTE_DSL_DISABLE_FILE_CACHING"] = "1"


from pathlib import Path
from typing import Any, Optional

import torch.cuda
from cutlass.cute.nvgpu.common import OpError
from cutlass._mlir.ir import MLIRError

from torch.cuda.nvtx import range as nvtx_range

from utils import set_seed, clear_l2_cache_large as clear_l2_cache

try:
    from task import TestSpec
except ImportError:
    TestSpec = dict

from reference import check_implementation, generate_input

NUM_ITERATIONS_PER_BENCHMARK = 15
UNSERIALIZABLE_EXCEPTIONS = (OpError, MLIRError)


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


def _combine(a: int, b: int) -> int:
    # combine two integers into one:
    # we need this to generate a secret seed based on the test-level seed and
    # the global secret seed.
    # the test-level seeds are public knowledge, and typically relatively small numbers,
    # so we need to make sure they don't provide any useful info for the full seed.
    # This Cantor construction ensures that if the secret seed is a large number,
    # then so is the overall seed.
    return int(a + (a + b) * (a + b + 1) // 2)


def get_test_cases(file_name: str, seed: Optional[int]) -> list[TestCase]:
    try:
        content = Path(file_name).read_text()
    except Exception as E:
        print(f"Could not open test file`{file_name}`: {E}", file=sys.stderr)
        exit(113)

    tests = []
    lines = content.splitlines()
    # Match key: value pairs where value can be:
    # - a list like [1, 2, 3] (needed for group gemm which has per-group dimensions)
    # - a tuple like (1, 2, 3)
    # - an integer
    # - an alphabetic string
    match = r"\s*([a-zA-Z_]+)\s*:\s*(\[[^\]]*\]|\([^)]*\)|[a-zA-Z_]+|[+-]?[0-9]+)\s*"
    for line in lines:
        parts = line.split(";")
        case = {}
        for part in parts:
            matched = re.match(match, part)
            if not re.fullmatch(match, part):
                print(f"invalid test case: '{line}': '{part}'", file=sys.stderr)
                exit(113)
            key = matched[1]
            val = matched[2]
            try:
                val = int(val)
            except ValueError:
                # Try parsing as tuple/list (e.g., [1, 2, 3] for group gemm dimensions)
                if (val.startswith("(") and val.endswith(")")) or (
                    val.startswith("[") and val.endswith("]")
                ):
                    try:
                        inner = val[1:-1].strip()
                        if inner:
                            val = tuple(int(x.strip()) for x in inner.split(","))
                        else:
                            val = tuple()
                    except ValueError:
                        pass

            case[key] = val
        tests.append(TestCase(spec=line, args=case))

    if seed is not None:
        for test in tests:
            if "seed" in test.args:
                test.args["seed"] = _combine(test.args["seed"], seed)

    return tests


@dataclasses.dataclass
class Stats:
    runs: int
    mean: float
    std: float
    err: float
    best: float
    worst: float


def calculate_stats(durations: list[int]):
    """
    Calculate statistical data from a list of durations.

    @param durations: A list of durations in nanoseconds.
    @return: A Stats object containing the number of runs, mean, standard deviation, error, best, and worst durations.
    """
    runs = len(durations)
    total = sum(durations)
    best = min(durations)
    worst = max(durations)

    avg = total / runs
    variance = sum(map(lambda x: (x - avg) ** 2, durations))
    std = math.sqrt(variance / (runs - 1))
    err = std / math.sqrt(runs)

    return Stats(
        runs=runs, mean=avg, std=std, err=err, best=float(best), worst=float(worst)
    )


def _clone_data(data):
    """
    Recursively goes through data and clones all tensors.
    """
    if isinstance(data, tuple):
        return tuple(_clone_data(x) for x in data)
    elif isinstance(data, list):
        return [_clone_data(x) for x in data]
    elif isinstance(data, dict):
        return {k: _clone_data(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.clone()
    else:
        return data


def _collect_output_tensors(output):
    """Collect tensors from nested output structure in deterministic order."""
    tensors = []

    def _walk(x):
        if isinstance(x, torch.Tensor):
            tensors.append(x)
        elif isinstance(x, (list, tuple)):
            for y in x:
                _walk(y)
        elif isinstance(x, dict):
            for k in sorted(x.keys()):
                _walk(x[k])

    _walk(output)
    return tensors


def _make_fingerprint_plan(output, gen, samples_per_tensor: int = 256):
    """
    Build a secret sampled hash plan for this output structure.
    """
    tensors = _collect_output_tensors(output)
    if not tensors:
        return []

    plan = []
    for t in tensors:
        n = int(t.numel())
        s = min(samples_per_tensor, n)
        if s <= 0:
            plan.append((0, None, None, None))
            continue
        idx = torch.randint(0, n, (s,), generator=gen, device=t.device, dtype=torch.int64)
        w1 = torch.randint(
            -(1 << 20), (1 << 20), (s,), generator=gen, device=t.device, dtype=torch.int32
        ).to(torch.float64)
        w2 = torch.randint(
            -(1 << 20), (1 << 20), (s,), generator=gen, device=t.device, dtype=torch.int32
        ).to(torch.float64)
        plan.append((n, idx, w1, w2))
    return plan


def _fingerprint_output(output, plan):
    """
    Compute a lightweight sampled fingerprint of output tensor contents.

    Returns two device scalars (h1, h2). If output changed post-return, the
    fingerprint almost certainly changes too.
    """
    tensors = _collect_output_tensors(output)
    if len(tensors) != len(plan):
        raise ValueError(
            f"output structure changed: expected {len(plan)} tensors, got {len(tensors)}"
        )

    if not tensors:
        z = torch.zeros((), dtype=torch.float64)
        return z, z

    device = tensors[0].device
    h1 = torch.zeros((), device=device, dtype=torch.float64)
    h2 = torch.zeros((), device=device, dtype=torch.float64)

    for t, (expected_n, idx, w1, w2) in zip(tensors, plan):
        n = int(t.numel())
        if n != expected_n:
            raise ValueError(f"output tensor size changed: expected {expected_n}, got {n}")
        if expected_n == 0:
            continue
        vals = t.reshape(-1).index_select(0, idx).to(torch.float64)
        vals = torch.nan_to_num(vals, nan=0.0, posinf=1e6, neginf=-1e6)
        h1 = h1 + (vals * w1).sum(dtype=torch.float64)
        h2 = h2 + (vals * w2).sum(dtype=torch.float64)
    return h1, h2


def _fingerprint_equal(a, b) -> bool:
    return torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def _run_single_test(test: TestCase):
    """
    Runs a single test case. Do not call directly
    """
    from submission import custom_kernel

    data = generate_input(**test.args)
    torch.cuda.synchronize()
    try:
        submission_output = custom_kernel(_clone_data(data))

    except UNSERIALIZABLE_EXCEPTIONS as E:
        print(f"Encountered {E}", file=sys.stderr)
        return False, str(E)
    torch.cuda.synchronize()
    return check_implementation(data, submission_output)


def run_single_test(pool: multiprocessing.Pool, test: TestCase):
    """
    Runs a single test in another process.
    """
    return pool.apply(_run_single_test, (test,))


def run_testing(
    logger: PopcornOutput, pool: multiprocessing.Pool, tests: list[TestCase]
):
    """
    Executes the actual test case code and checks for correctness.

    @param logger: A PopcornOutput object used for logging test results.
    @param tests: A list of TestCase objects representing the test cases to be executed.
    @return: An integer representing the exit status: 0 if all tests pass, otherwise 112.
    """
    passed = True
    logger.log("test-count", len(tests))
    for idx, test in enumerate(tests):
        logger.log(f"test.{idx}.spec", test.spec)
        good, message = run_single_test(pool, test)
        if not good:
            logger.log(f"test.{idx}.status", "fail")
            logger.log(f"test.{idx}.error", message)
            passed = False
        else:
            logger.log(f"test.{idx}.status", "pass")
            if message:
                logger.log(f"test.{idx}.message", message)

    if passed:
        logger.log("check", "pass")
        return 0
    else:
        logger.log("check", "fail")
        return 112


def _run_single_benchmark(
    test: TestCase, recheck: bool, max_repeats: int, max_time_ns: float
) -> Stats | Any:
    """
    Runs one benchmark. Do not call directly.
    """
    from submission import custom_kernel

    durations = []
    data_list = []
    # generate input data once

    local_seed = test.args.get("seed", None)
    for i in range(NUM_ITERATIONS_PER_BENCHMARK):
        if local_seed is not None:
            local_seed += 42
            args = {**test.args, "seed": local_seed}
        else:
            args = test.args
        data = generate_input(**args)
        data_list.append(data)

    check_copy = _clone_data(data_list)
    # Deterministic but hidden probe stream.
    # In benchmark mode we use randomized call windows and sparse probes.
    # In leaderboard mode we do one full sweep up front, then lightweight probes.
    probe_seed = _combine(int(test.args.get("seed", 0) or 0), 0x4D455452)
    probe_rng = random.Random(probe_seed)
    full_calls = len(data_list)
    fp_gen = torch.Generator(device="cuda")
    fp_seed = _combine(probe_seed, 0xF1A9E5) & ((1 << 63) - 1)
    fp_gen.manual_seed(fp_seed)

    # First, one obligatory correctness check on fresh clones.
    outputs = []
    try:
        for data in data_list:
            output = custom_kernel(_clone_data(data))
            outputs.append(output)
    except UNSERIALIZABLE_EXCEPTIONS as E:
        return f"Encountered {E}"
    for reference_output, custom_output in zip(check_copy, outputs):
        good, message = check_implementation(reference_output, custom_output)
        if not good:
            return message
    try:
        fingerprint_plans = [_make_fingerprint_plan(out, fp_gen) for out in outputs]
    except Exception as E:
        return f"fingerprint plan build failed: {E}"

    # Timing: per-call intervals captured with CUDA events and one sync.
    # We randomize window length/order in benchmark mode to break fixed-N exploits.
    # Data is cloned each iteration to prevent object-identity caching.

    bm_start_time = time.perf_counter_ns()
    for i in range(max_repeats):
        # Clone and shuffle data before timing to prevent both
        # object-identity caching and call-order caching exploits
        iteration_data = _clone_data(data_list)
        shuffle_order = list(range(len(iteration_data)))
        random.shuffle(shuffle_order)
        iteration_data = [iteration_data[j] for j in shuffle_order]

        torch.cuda.synchronize()

        if recheck:
            integrity_repeat = (i == 0) or (i % 20 == 0)
        else:
            integrity_repeat = (i < 3) or (i % 25 == 0)

        if recheck:
            call_indices = list(range(full_calls))
        else:
            call_indices = list(range(full_calls))
            probe_rng.shuffle(call_indices)
            if integrity_repeat:
                # Integrity repeats must exercise the full call window so
                # flush-at-N exploits cannot hide behind short random windows.
                n_calls = full_calls
            else:
                min_calls = max(4, full_calls - 6)
                n_calls = probe_rng.randint(min_calls, full_calls)
            call_indices = call_indices[:n_calls]

        outputs = []
        events = [torch.cuda.Event(enable_timing=True) for _ in range(len(call_indices) + 1)]

        if integrity_repeat and len(call_indices) <= 1:
            in_loop_probe_pos = 0 if call_indices else None
        elif integrity_repeat:
            # Probe before last call to expose deferred-until-last behavior.
            in_loop_probe_pos = probe_rng.randrange(0, len(call_indices) - 1)
        else:
            in_loop_probe_pos = None

        probe_snapshot = None
        events[0].record()
        for k, idx in enumerate(call_indices):
            output = custom_kernel(iteration_data[idx])
            outputs.append((idx, output))
            events[k + 1].record()

            # Snapshot output state immediately after return; compare again after
            # the full window to detect post-return deferred writes.
            if in_loop_probe_pos is not None and k == in_loop_probe_pos:
                try:
                    fp_before = _fingerprint_output(output, fingerprint_plans[idx])
                except Exception as E:
                    return f"fingerprint snapshot failed: {E}"
                probe_snapshot = (idx, output, fp_before)
        torch.cuda.synchronize()

        if probe_snapshot is not None:
            idx, probe_output, fp_before = probe_snapshot
            try:
                fp_after = _fingerprint_output(probe_output, fingerprint_plans[idx])
            except Exception as E:
                return f"fingerprint verify failed: {E}"
            torch.cuda.synchronize()
            if not _fingerprint_equal(fp_before, fp_after):
                return (
                    "detected deferred/cross-call output mutation "
                    f"(call_index={idx}, window_calls={len(call_indices)})"
                )

        per_call_durations = [
            events[k].elapsed_time(events[k + 1]) * 1e6 for k in range(len(call_indices))
        ]

        # Correctness policy:
        # - benchmark: sparse hidden integrity repeats + randomized windows/order.
        # - leaderboard: sparse integrity repeats; first repeat gets full sweep.
        if recheck:
            if i == 0:
                check_positions = list(range(len(outputs)))
            else:
                check_positions = []
        else:
            check_positions = []

        for pos in check_positions:
            idx, output = outputs[pos]
            good, message = check_implementation(check_copy[idx], output)
            if not good:
                return message

        duration = sum(per_call_durations) / len(call_indices)
        if not integrity_repeat:
            durations.append(duration)

        total_bm_duration = time.perf_counter_ns() - bm_start_time
        if (
            len(durations) > 1 and total_bm_duration > 1e8
        ):  # at least 2 runs, and at least 100 ms total time
            stats = calculate_stats(durations)
            # stop if either
            # a) relative error dips below 0.1%
            # b) we exceed the total time limit for benchmarking the kernel
            # c) we exceed 2 minutes of total wallclock time.
            if (
                stats.err / stats.mean < 0.001
                or stats.mean * stats.runs > max_time_ns
                or total_bm_duration > 120e9
            ):
                break

    if not durations:
        return "benchmark produced no timing samples"

    return calculate_stats(durations)


def run_single_benchmark(
    pool: multiprocessing.Pool,
    test: TestCase,
    recheck: bool,
    max_repeats: int,
    max_time_ns: float,
):
    """
    For a particular test case, check correctness (if applicable) and grab runtime results.

    @param pool: Process on which the benchmark will be launched.
    @param test: TestCase object.
    @param recheck: Flag for whether to explicitly check functional correctness.
    @param max_repeats: Number of trials to repeat.
    @param max_time_ns: Timeout time in nanoseconds.
    @return: A Stats object for this particular benchmark case or an error if the test fails.
    """
    return pool.apply(_run_single_benchmark, (test, recheck, max_repeats, max_time_ns))


def run_benchmarking(
    logger: PopcornOutput, pool: multiprocessing.Pool, tests: list[TestCase]
):
    """
    Executes benchmarking code for a CUDA Kernel and logs runtimes.

    @param logger: A PopcornOutput object used for logging benchmark results.
    @param pool: Process on which the benchmarks will be launched.
    @param tests: A list of TestCase objects representing the test cases to be benchmarked.
    @return: An integer representing the exit status: 0 if all benchmarks pass, otherwise 112.
    """

    run_single_benchmark(pool, tests[0], False, 100, 10e7)

    passed = True
    logger.log("benchmark-count", len(tests))
    for idx, test in enumerate(tests):
        logger.log(f"benchmark.{idx}.spec", test.spec)
        result = run_single_benchmark(pool, test, False, 100, 10e9)
        if isinstance(result, Stats):
            for field in dataclasses.fields(Stats):
                logger.log(f"benchmark.{idx}.{field.name}", getattr(result, field.name))
        else:
            passed = False
            logger.log(f"benchmark.{idx}.status", "fail")
            logger.log(f"benchmark.{idx}.error", result)

    if passed:
        logger.log("check", "pass")
        return 0
    else:
        logger.log("check", "fail")
        return 112


def _run_single_profile_torch(test: TestCase) -> str:
    """
    Profiles a single benchmark using the torch profiler.
    """
    from submission import custom_kernel
    from torch.profiler import profile, ProfilerActivity

    with nvtx_range("generate input"):
        data = generate_input(**test.args)
        torch.cuda.synchronize()

    cloned = _clone_data(data)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with nvtx_range("custom_kernel"):
            submission_output = custom_kernel(cloned)
            torch.cuda.synchronize()

    return prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)


def _run_single_profile_ncu(test: TestCase) -> str:
    """
    Profiles a single benchmark using ncu. Note: this does not
    invoke NCU; instead, it is expected that eval is launched
    under NCU, and this function will rurnthe kernel excactly
    once in the 'custom_kernel' nvtx range.
    """
    from submission import custom_kernel

    with nvtx_range("generate input"):
        data = generate_input(**test.args)
        torch.cuda.synchronize()

    cloned = _clone_data(data)
    with nvtx_range("custom_kernel"):
        submission_output = custom_kernel(cloned)
        torch.cuda.synchronize()

    return ""


def _combine_traces(traces: list["EventList"]) -> "EventList":
    """
    Combine multiple event traces obtained from multiple (distributed) torch.profiler
    activities. This function simply aggregates the data as like `prof.key_averages()`,
    except over multiple traces. Most of this function is reimplemented
    from `torch.autograd.profiler_util.EventList.key_averages()`.
    """
    from torch.autograd.profiler_util import FunctionEventAvg, EventList
    from collections import defaultdict

    def get_key(event) -> tuple[str, ...]:
        return (
            str(event.key),
            str(event.node_id),
            str(event.device_type),
            str(event.is_legacy),
            str(event.is_user_annotation),
        )

    stats: dict[tuple[str, ...], FunctionEventAvg] = defaultdict(FunctionEventAvg)

    for events in traces:
        for event in events:
            stats[get_key(event)].add(event)

    avg_list = EventList(stats.values())
    for event in avg_list:
        event.stack = []
        event.input_shapes = ""
        event.overload_name = ""

    return avg_list


def run_single_profile(test: TestCase, pool: multiprocessing.Pool) -> str:
    """
    Runs a single profiling activity in another process.
    """
    if bool(os.getenv("POPCORN_NCU", "0")):
        return pool.apply(_run_single_profile_ncu, (test,))
    else:
        return pool.apply(_run_single_profile_torch, (test,))


def run_profiling(
    logger: PopcornOutput, pool: multiprocessing.Pool, tests: list[TestCase]
):
    logger.log("benchmark-count", len(tests))
    for idx, test in enumerate(tests):
        logger.log(f"benchmark.{idx}.spec", test.spec)
        report = run_single_profile(test, pool)
        logger.log(
            f"benchmark.{idx}.report",
            base64.b64encode(report.encode("utf-8"), b"+*").decode("utf-8"),
        )
    logger.log("check", "pass")
    return 0


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
        import multiprocessing

        mp_context = multiprocessing.get_context("spawn")
        with mp_context.Pool(1, initializer=_init_worker) as pool:
            if mode == "test":
                return run_testing(logger, pool, tests)
            if mode == "benchmark":
                return run_benchmarking(logger, pool, tests)

            if mode == "leaderboard":
                # Warmup all test shapes to ensure consistent benchmarking
                for test in tests:
                    run_single_benchmark(pool, test, False, 50, 5e8)
                logger.log("benchmark-count", len(tests))
                passed = True
                for i in range(len(tests)):
                    result = run_single_benchmark(pool, tests[i], True, 100, 30e9)
                    logger.log(f"benchmark.{i}.spec", tests[i].spec)
                    if isinstance(result, Stats):
                        for field in dataclasses.fields(Stats):
                            logger.log(
                                f"benchmark.{i}.{field.name}",
                                getattr(result, field.name),
                            )
                    else:
                        passed = False
                        logger.log(f"benchmark.{i}.status", "fail")
                        logger.log(
                            f"benchmark.{i}.error", str(result)
                        )  # TODO: Make sure result implements __str__?
                        break

                logger.log("check", "pass" if passed else "fail")
                return 0 if passed else 112
            elif mode == "profile":
                return run_profiling(logger, pool, tests)
            else:
                # TODO: Implement script mode
                return 2


if __name__ == "__main__":
    sys.exit(main())
