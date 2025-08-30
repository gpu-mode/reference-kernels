import base64
import copy
import dataclasses
import multiprocessing
import re
import time
import os
import sys
import math
from pathlib import Path
from typing import Any, Optional

import torch.cuda

from utils import set_seed, clear_l2_cache

try:
    from task import TestSpec
except ImportError:
    TestSpec = dict

from reference import check_implementation, generate_input


class PopcornOutput:
    def __init__(self, fd: int):
        self.file = os.fdopen(fd, 'w')
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
    match = r"\s*([a-zA-Z_]+):\s*([a-zA-Z]+|[+-]?[0-9]+)\s*"
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

    return Stats(runs=runs, mean=avg, std=std, err=err, best=float(best),
                 worst=float(worst))


def _clone_data(data, rank: int):
    """
    Recursively goes through data and clones all tensors.
    """
    if isinstance(data, tuple):
        return tuple(_clone_data(x, rank) for x in data)
    elif isinstance(data, list):
        return [_clone_data(x, rank) for x in data]
    elif isinstance(data, dict):
        return {k: _clone_data(v, rank) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        device = f"cuda:{rank}"
        return data.clone().to(device)
    else:
        return data


def wrap_check_implementation(data, submission_output):
    # Old version returned just a single string, new version
    # returns (bool, str); this function ensures compatibility with old
    # problem definitions.
    result = check_implementation(data, submission_output)
    if isinstance(result, tuple):
        return result
    else:
        return not bool(result), result


def _run_single_test(test: TestCase):
    """
    Runs a single test case. Do not call directly
    """
    from submission import custom_kernel
    data = generate_input(**test.args)
    torch.cuda.synchronize()
    submission_output = custom_kernel(_clone_data(data, 0))
    torch.cuda.synchronize()
    return wrap_check_implementation(data, submission_output)


def _run_distributed_test(test: TestCase, rank: int):
    """
    Runs a single test case. Do not call directly
    """
    from submission import custom_kernel
    import torch.distributed as dist
    world_size = test.args["world_size"]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size, device_id=torch.device(f'cuda:{rank}'))
    try:
        data = generate_input(**test.args, rank=rank)
        torch.cuda.synchronize()
        submission_output = custom_kernel(_clone_data(data, rank))
        torch.cuda.synchronize()
        return wrap_check_implementation(data, submission_output)
    finally:
        dist.destroy_process_group()


def run_multi_gpu_test(pool: multiprocessing.Pool, test: TestCase, world_size: int):
    """
    Runs a single test in another process.
    """
    rets = []
    # world_size is a mandatory argument for multi-gpu tests
    for i in range(world_size):
        rets.append(
            pool.apply_async(
                _run_distributed_test,
                args=(test, i),
            )
        )
    # 60 seconds should be more than enough, we want tests to be fast
    rets = [el.get(60) for el in rets]

    correct = all(ret[0] for ret in rets)
    error_messages = str.join("\n", [f"rank {rank} - {ret[1]}" for rank, ret in enumerate(rets) if not ret[0]])
    return correct, error_messages


def run_single_test(pool: multiprocessing.Pool, test: TestCase):
    """
    Runs a single test in another process.
    """
    world_size = test.args.get("world_size", None)
    if world_size is None:
        return pool.apply(_run_single_test, (test,))
    else:
        return run_multi_gpu_test(pool, test, world_size)


def run_testing(logger: PopcornOutput, pool: multiprocessing.Pool, tests: list[TestCase]):
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


def _run_single_benchmark(test: TestCase, recheck: bool, max_repeats: int, max_time_ns: float) -> Stats | Any:
    """
    Runs one benchmark. Do not call directly.
    """
    from submission import custom_kernel

    durations = []
    # generate input data once
    data = generate_input(**test.args)
    check_copy = _clone_data(data, 0)
    #  first, one obligatory correctness check
    output = custom_kernel(data)
    good, message = wrap_check_implementation(check_copy, output)
    if not good:
        return message

    # now, do multiple timing runs without further correctness testing
    # there is an upper bound of 100 runs, and a lower bound of 3 runs;
    # otherwise, we repeat until we either measure at least 10 full seconds,
    # or the relative error of the mean is below 1%.

    bm_start_time = time.perf_counter_ns()
    for i in range(max_repeats):
        if recheck:
            # ensure we use a different seed for every benchmark
            if "seed" in test.args:
                test.args["seed"] += 13

            data = generate_input(**test.args)
            check_copy = _clone_data(data, 0)
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        clear_l2_cache()

        start_event.record()
        output = custom_kernel(data)
        end_event.record()
        torch.cuda.synchronize()
        duration = start_event.elapsed_time(end_event) * 1e6  # Convert ms to ns

        if recheck:
            good, message = check_implementation(check_copy, output)
            if not good:
                return message

        del output
        durations.append(duration)

        if i > 1:
            total_bm_duration = time.perf_counter_ns() - bm_start_time
            stats = calculate_stats(durations)
            # stop if either
            # a) relative error dips below 0.1%
            # b) we exceed the total time limit for benchmarking the kernel
            # c) we exceed 2 minutes of total wallclock time.
            if stats.err / stats.mean < 0.001 or stats.mean * stats.runs > max_time_ns or total_bm_duration > 120e9:
                break

    return calculate_stats(durations)


def _run_distributed_benchmark(test: TestCase, rank: int, recheck: bool, max_repeats: int,
                               max_time_ns: float) -> Stats | Any:
    """
    Runs one distributed benchmark. Do not call directly.
    """
    from submission import custom_kernel
    import torch.distributed as dist

    world_size = test.args["world_size"]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size, device_id=torch.device(f'cuda:{rank}'))

    try:
        durations = []
        # generate input data once
        data = generate_input(**test.args, rank=rank)
        check_copy = _clone_data(data, rank)

        # first, one obligatory correctness check
        output = custom_kernel(_clone_data(data, rank))
        good, message = wrap_check_implementation(check_copy, output)
        if not good:
            return message

        # now, do multiple timing runs with proper distributed synchronization
        bm_start_time = time.perf_counter_ns()
        for i in range(max_repeats):
            error_message = None
            if recheck:
                # ensure we use a different seed for every benchmark
                if "seed" in test.args:
                    test.args["seed"] += 13

                data = generate_input(**test.args, rank=rank)
                check_copy = _clone_data(data, rank)

            # Synchronize all ranks before timing
            clear_l2_cache()
            torch.cuda.synchronize()
            dist.barrier()

            # Use distributed timing - only rank 0 records the overall time
            if rank == 0:
                start_time = time.perf_counter_ns()

            # All ranks execute the kernel
            output = custom_kernel(_clone_data(data, rank))

            # Synchronize all ranks after kernel execution
            torch.cuda.synchronize()
            dist.barrier()

            if rank == 0:
                end_time = time.perf_counter_ns()
                duration = end_time - start_time  # Already in nanoseconds
                durations.append(duration)

            if recheck:
                good, message = check_implementation(check_copy, output)
                if not good:
                    error_message = message

            del output

            has_error = torch.tensor(1 if error_message is not None else 0, dtype=torch.int32, device=f'cuda:{rank}')
            dist.reduce(has_error, 0)
            if has_error.item() > 0:
                return error_message

            # Only rank 0 checks convergence criteria
            if rank == 0 and i > 1:
                total_bm_duration = time.perf_counter_ns() - bm_start_time
                stats = calculate_stats(durations)
                # stop if either
                # a) relative error dips below 0.1%
                # b) we exceed the total time limit for benchmarking the kernel
                # c) we exceed 2 minutes of total wallclock time.
                should_stop = (stats.err / stats.mean < 0.001 or
                               stats.mean * stats.runs > max_time_ns or
                               total_bm_duration > 120e9)
            else:
                should_stop = False

            # Broadcast stop decision to all ranks
            stop_tensor = torch.tensor(should_stop, dtype=torch.bool, device=f'cuda:{rank}')
            dist.broadcast(stop_tensor, 0)

            if stop_tensor.item():
                break

        # Only rank 0 returns meaningful stats
        if rank == 0:
            return calculate_stats(durations)
        else:
            # Non-zero ranks return a dummy stats object
            return Stats(runs=len(durations), mean=0.0, std=0.0, err=0.0, best=0.0, worst=0.0)

    finally:
        dist.destroy_process_group()


def run_multi_gpu_benchmark(pool: multiprocessing.Pool, test: TestCase, recheck: bool, max_repeats: int,
                            max_time_ns: float, world_size: int):
    """
    Runs a multi-GPU benchmark across all ranks.
    """
    rets = []
    for i in range(world_size):
        rets.append(
            pool.apply_async(
                _run_distributed_benchmark,
                args=(test, i, recheck, max_repeats, max_time_ns),
            )
        )

    # 120 seconds for benchmarking + we run a pre-benchmark test and want to leave some slack
    rets = [el.get(timeout=180) for el in rets]

    # For multi-GPU benchmarking, only rank 0 has meaningful stats
    failed_ranks = []
    rank_0_result = None

    for rank, ret in enumerate(rets):
        if isinstance(ret, Stats):
            if rank == 0:
                rank_0_result = ret
        else:
            # ret is an error message
            failed_ranks.append((rank, ret))

    if failed_ranks:
        error_messages = str.join("\n", [f"rank {rank} - {msg}" for rank, msg in failed_ranks])
        return error_messages
    else:
        return rank_0_result if rank_0_result else "No stats returned from rank 0"


def run_single_benchmark(pool: multiprocessing.Pool, test: TestCase, recheck: bool, max_repeats: int,
                         max_time_ns: float):
    """
    For a particular test case, check correctness (if applicable) and grab runtime results.

    @param pool: Process on which the benchmark will be launched.
    @param test: TestCase object.
    @param recheck: Flag for whether to explicitly check functional correctness.
    @param max_repeats: Number of trials to repeat.
    @param max_time_ns: Timeout time in nanoseconds.
    @return: A Stats object for this particular benchmark case or an error if the test fails.
    """

    world_size: Optional[int] = test.args.get("world_size", None)
    if world_size is None:
        return pool.apply(_run_single_benchmark, (test, recheck, max_repeats, max_time_ns))
    else:
        return run_multi_gpu_benchmark(pool, test, recheck, max_repeats, max_time_ns, world_size)


def run_benchmarking(logger: PopcornOutput, pool: multiprocessing.Pool, tests: list[TestCase]):
    """
    Executes benchmarking code for a CUDA Kernel and logs runtimes.

    @param logger: A PopcornOutput object used for logging benchmark results.
    @param pool: Process on which the benchmarks will be launched.
    @param tests: A list of TestCase objects representing the test cases to be benchmarked.
    @return: An integer representing the exit status: 0 if all benchmarks pass, otherwise 112.
    """
    # warm up
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


def _run_single_profile(test: TestCase) -> str:
    """
    Runs a single test case. Do not call directly
    """
    from submission import custom_kernel
    from torch.profiler import profile, ProfilerActivity
    data = generate_input(**test.args)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        submission_output = custom_kernel(_clone_data(data, 0))
        torch.cuda.synchronize()

    return prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)


def _run_distributed_profile(test: TestCase, rank: int) -> "EventList":
    """
    Runs a single profiling case. Do not call directly
    """
    from submission import custom_kernel
    from torch.profiler import profile, ProfilerActivity
    import torch.distributed as dist
    world_size = test.args["world_size"]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size, device_id=torch.device(f'cuda:{rank}'))

    try:
        data = generate_input(**test.args, rank=rank)
        data = _clone_data(data, rank)
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            submission_output = custom_kernel(data)
            torch.cuda.synchronize()

        return prof.events()

    finally:
        dist.destroy_process_group()


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


def run_multi_gpu_profile(pool: multiprocessing.Pool, test: TestCase, world_size: int) -> str:
    """
    Runs a single test in another process.
    """
    rets = []
    # world_size is a mandatory argument for multi-gpu tests
    for i in range(world_size):
        rets.append(
            pool.apply_async(
                _run_distributed_profile,
                args=(test, i),
            )
        )

    rets = [el.get(120) for el in rets]
    return _combine_traces(rets).table(sort_by="self_cuda_time_total", row_limit=20)


def run_single_profile(test: TestCase, pool: multiprocessing.Pool) -> str:
    """
    Runs a single profiling activity in another process.
    """
    world_size = test.args.get("world_size", None)
    if world_size is None:
        return pool.apply(_run_single_profile, (test,))
    else:
        return run_multi_gpu_profile(pool, test, world_size)


def run_profiling(logger: PopcornOutput, pool: multiprocessing.Pool, tests: list[TestCase]):
    logger.log("benchmark-count", len(tests))
    for idx, test in enumerate(tests):
        logger.log(f"benchmark.{idx}.spec", test.spec)
        report = run_single_profile(test, pool)
        logger.log(f"benchmark.{idx}.report", base64.b64encode(report.encode("utf-8"), b"+*").decode("utf-8"))
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
    n_gpus = int(os.getenv("POPCORN_GPUS", "1"))
    seed = int(seed) if seed else None
    set_seed(seed or 42)
    tests = get_test_cases(sys.argv[2], seed)

    with PopcornOutput(int(fd)) as logger:
        import multiprocessing
        mp_context = multiprocessing.get_context('spawn')
        with mp_context.Pool(n_gpus) as pool:
            if mode == "test":
                return run_testing(logger, pool, tests)
            if mode == "benchmark":
                return run_benchmarking(logger, pool, tests)

            if mode == "leaderboard":
                # warmup
                run_single_benchmark(pool, tests[0], False, 100, 1e7)
                logger.log("benchmark-count", len(tests))
                passed = True
                for i in range(len(tests)):
                    result = run_single_benchmark(pool, tests[i], True, 100, 30e9)
                    logger.log(f"benchmark.{i}.spec", tests[i].spec)
                    if isinstance(result, Stats):
                        for field in dataclasses.fields(Stats):
                            logger.log(f"benchmark.{i}.{field.name}", getattr(result, field.name))
                    else:
                        passed = False
                        logger.log(f"benchmark.{i}.status", "fail")
                        logger.log(f"benchmark.{i}.error", str(result))  # TODO: Make sure result implements __str__?
                        break

                logger.log("check", "pass" if passed else "fail")
            elif mode == "profile":
                run_profiling(logger, pool, tests)
            else:
                # invalid mode
                return 2


if __name__ == "__main__":
    sys.exit(main())
