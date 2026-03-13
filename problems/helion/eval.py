import base64
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
    return int(a + (a+b)*(a+b+1)//2)


def get_test_cases(file_name: str, seed: Optional[int]) -> list[TestCase]:
    try:
        content = Path(file_name).read_text()
    except Exception as E:
        print(f"Could not open test file`{file_name}`: {E}", file=sys.stderr)
        exit(113)

    tests = []
    lines = content.splitlines()
    match = r"\s*([a-zA-Z_]\w*):\s*([a-zA-Z_]\w*|[+-]?[0-9]+)\s*"
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
                if val == "true":
                    val = True
                elif val == "false":
                    val = False

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
    variance = sum(map(lambda x: (x - avg)**2, durations))
    std = math.sqrt(variance / (runs - 1))
    err = std / math.sqrt(runs)

    return Stats(runs=runs, mean=avg, std=std, err=err, best=float(best),
                 worst=float(worst))


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


def _copy_data_inplace(dst, src):
    """
    Recursively copy tensor data from src into dst (same structure, same shapes).
    Used to feed new inputs into CUDA graph buffers without recapturing.
    """
    if isinstance(dst, torch.Tensor):
        dst.copy_(src)
    elif isinstance(dst, (tuple, list)):
        for d, s in zip(dst, src):
            _copy_data_inplace(d, s)
    elif isinstance(dst, dict):
        for k in dst:
            _copy_data_inplace(dst[k], src[k])


def _do_bench_cudagraph(fn, rep_ms=100, return_mode="mean", clear_l2=True):
    """
    Benchmark fn using CUDA graphs with optional L2 cache clearing.
    Based on triton.testing.do_bench_cudagraph + triton-lang/triton#8384.

    :param fn: Callable to benchmark (no args).
    :param rep_ms: Target repetition time per measurement in milliseconds.
    :param return_mode: "min", "max", "mean", "median", or "all" (list of ms).
    :param clear_l2: If True, flush L2 cache before each invocation and subtract
                     the flushing overhead from reported times.
    :return: Time(s) in milliseconds.
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    # 256 MB cache tensor — larger than any current GPU L2
    cache = torch.empty(32 * 1024 * 1024, dtype=torch.int64, device="cuda") if clear_l2 else None

    def maybe_clear_cache():
        if cache is not None:
            cache.zero_()

    with torch.cuda.stream(torch.cuda.Stream()):
        # warmup
        maybe_clear_cache()
        fn()

        # step 1 — estimate per-call time
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            maybe_clear_cache()
            fn()
        end_event.record()
        torch.cuda.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5

        n_repeat = max(1, int(rep_ms / estimate_ms)) if estimate_ms > 0 else 1000

        # step 2 — capture graph with n_repeat unrolled calls
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n_repeat):
                maybe_clear_cache()
                fn()
        torch.cuda.synchronize()

        # step 3 — if L2 clearing enabled, capture a separate graph to measure
        # the clearing overhead so we can subtract it
        cache_clear_graph = None
        if clear_l2:
            cache_clear_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cache_clear_graph):
                for _ in range(n_repeat):
                    maybe_clear_cache()
            torch.cuda.synchronize()

        # step 4 — measure
        n_retries = 10
        cache_clear_times = []
        total_times = []
        for _ in range(n_retries):
            if cache_clear_graph is not None:
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                cache_clear_graph.replay()
                e.record()
                torch.cuda.synchronize()
                cache_clear_times.append(s.elapsed_time(e) / n_repeat)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            g.replay()
            end_event.record()
            torch.cuda.synchronize()
            total_times.append(start_event.elapsed_time(end_event) / n_repeat)

        if clear_l2:
            ret = [max(0, t - c) for t, c in zip(total_times, cache_clear_times)]
        else:
            ret = total_times

        if return_mode == "all":
            return ret
        elif return_mode == "min":
            return min(ret)
        elif return_mode == "max":
            return max(ret)
        elif return_mode == "mean":
            return sum(ret) / len(ret)
        elif return_mode == "median":
            return sorted(ret)[len(ret) // 2]


def _run_single_test(test: TestCase):
    """
    Runs a single test case via CUDA graph capture + replay.
    This validates that the kernel is capturable and produces correct output.
    """
    from submission import custom_kernel
    from reference import check_implementation, generate_input

    data = generate_input(**test.args)
    check_copy = _clone_data(data)

    # Warmup call to trigger JIT compilation (outside graph capture)
    _ = custom_kernel(_clone_data(data))
    torch.cuda.synchronize()

    # Capture and replay through CUDA graph
    input_data = _clone_data(data)
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            output = custom_kernel(input_data)
    except Exception as e:
        return False, f"Failed to capture kernel in CUDA graph: {e}"
    g.replay()
    torch.cuda.synchronize()

    return check_implementation(check_copy, output)


def run_single_test(pool: multiprocessing.Pool, test: TestCase):
    """
    Runs a single test in another process.
    """
    return pool.apply(_run_single_test, (test,))


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


def _run_single_benchmark(test: TestCase, recheck: bool, rep_ms: int) -> Stats | Any:
    """
    Runs one benchmark. Do not call directly.

    Correctness is verified via CUDA graph capture + replay first.
    Timing only runs if all correctness checks pass.

    :param test: Test case with input arguments.
    :param recheck: If True, run additional correctness checks with varying seeds.
    :param rep_ms: Target repetition time per measurement in milliseconds.
    """
    from submission import custom_kernel
    from reference import check_implementation, generate_input

    data = generate_input(**test.args)
    check_copy = _clone_data(data)

    # Warmup (JIT compilation)
    _ = custom_kernel(_clone_data(data))
    torch.cuda.synchronize()

    # Capture in CUDA graph and run initial correctness check
    input_data = _clone_data(data)
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            output = custom_kernel(input_data)
    except Exception as e:
        return f"Failed to capture kernel in CUDA graph: {e}"
    g.replay()
    torch.cuda.synchronize()
    good, message = check_implementation(check_copy, output)
    if not good:
        return message

    if recheck:
        # Reuse the captured graph with new input data for each seed
        for i in range(10):
            if "seed" in test.args:
                test.args["seed"] += 13
            new_data = generate_input(**test.args)
            check_copy = _clone_data(new_data)
            _copy_data_inplace(input_data, new_data)
            g.replay()
            torch.cuda.synchronize()
            good, message = check_implementation(check_copy, output)
            if not good:
                return message

    # Timing (only reached if all correctness checks passed)
    data = generate_input(**test.args)
    fn = lambda: custom_kernel(data)
    times_ms = _do_bench_cudagraph(fn, rep_ms=rep_ms, return_mode="all", clear_l2=True)
    time.sleep(10)  # GPU cooldown to avoid thermal throttling
    durations = [t * 1e6 for t in times_ms]  # convert ms to ns
    return calculate_stats(durations)


def run_single_benchmark(pool: multiprocessing.Pool, test: TestCase, recheck: bool, rep_ms: int):
    """
    Run a benchmark in a subprocess.

    :param pool: Process pool.
    :param test: TestCase object.
    :param recheck: Flag for whether to explicitly check functional correctness.
    :param rep_ms: Target repetition time per measurement in milliseconds.
    :return: A Stats object or an error string.
    """
    return pool.apply(_run_single_benchmark, (test, recheck, rep_ms))


def run_benchmarking(logger: PopcornOutput, pool: multiprocessing.Pool, tests: list[TestCase]):
    """
    Executes benchmarking code for a CUDA Kernel and logs runtimes.

    @param logger: A PopcornOutput object used for logging benchmark results.
    @param pool: Process on which the benchmarks will be launched.
    @param tests: A list of TestCase objects representing the test cases to be benchmarked.
    @return: An integer representing the exit status: 0 if all benchmarks pass, otherwise 112.
    """
    # warm up
    run_single_benchmark(pool, tests[0], False, 20)

    passed = True
    logger.log("benchmark-count", len(tests))
    for idx, test in enumerate(tests):
        logger.log(f"benchmark.{idx}.spec", test.spec)
        result = run_single_benchmark(pool, test, False, 100)
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


def run_single_profile(test: TestCase) -> str:
    """
    Runs a single test case. Do not call directly
    """
    from submission import custom_kernel
    from reference import generate_input
    from torch.profiler import profile, record_function, ProfilerActivity
    data = generate_input(**test.args)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        submission_output = custom_kernel(_clone_data(data))
        torch.cuda.synchronize()
    return prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)


def run_profiling(logger: PopcornOutput, tests: list[TestCase]):
    logger.log("benchmark-count", len(tests))
    for idx, test in enumerate(tests):
        logger.log(f"benchmark.{idx}.spec", test.spec)
        report = run_single_profile(test)
        logger.log(f"benchmark.{idx}.report", base64.b64encode(report.encode("utf-8"), b"+*").decode("utf-8"))
    logger.log("check", "pass")
    return 0


def run_local():
    """
    Local eval mode: reads task.yml from a problem directory, runs correctness tests
    and benchmarks, prints results to stdout. No Popcorn infrastructure needed.

    Usage: python eval.py <mode> <problem_dir>
      mode: test, benchmark, or both
      problem_dir: path to the problem directory containing task.yml
    """
    import yaml

    if len(sys.argv) < 3:
        print("Usage: python eval.py <mode> <problem_dir>", file=sys.stderr)
        print("  mode: test, benchmark, or both", file=sys.stderr)
        print("  problem_dir: path to problem directory containing task.yml", file=sys.stderr)
        return 1

    mode = sys.argv[1]
    problem_dir = Path(sys.argv[2])

    if mode not in ("test", "benchmark", "both"):
        print(f"Unknown mode '{mode}'. Use 'test', 'benchmark', or 'both'.", file=sys.stderr)
        return 1

    problem_dir = problem_dir.resolve()
    task_path = problem_dir / "task.yml"
    if not task_path.exists():
        print(f"Error: task.yml not found in {problem_dir}", file=sys.stderr)
        return 1

    task = yaml.safe_load(task_path.read_text())

    # chdir into the problem directory so that `from submission import ...` works
    os.chdir(problem_dir)
    sys.path.insert(0, str(problem_dir))

    from utils import set_seed

    set_seed(42)
    exit_code = 0

    # --- Correctness tests ---
    if mode in ("test", "both"):
        tests = [TestCase(args=dict(t), spec=str(t)) for t in task.get("tests", [])]
        print(f"Running {len(tests)} correctness tests...")
        all_passed = True
        for idx, test in enumerate(tests):
            good, message = _run_single_test(test)
            status = "PASS" if good else "FAIL"
            print(f"  Test {idx}: {status}  {test.spec}")
            if not good:
                print(f"           {message}")
                all_passed = False
        if all_passed:
            print("All tests passed.")
        else:
            print("Some tests FAILED.")
            exit_code = 1

    # --- Benchmarks ---
    if mode in ("benchmark", "both"):
        benchmarks = [TestCase(args=dict(t), spec=str(t)) for t in task.get("benchmarks", [])]
        print(f"\nRunning {len(benchmarks)} benchmarks...")

        # Warmup
        _run_single_benchmark(benchmarks[0], False, 20)

        for idx, bench in enumerate(benchmarks):
            result = _run_single_benchmark(bench, False, 100)
            if isinstance(result, Stats):
                mean_ms = result.mean / 1e6  # Stats stores ns
                min_ms = result.best / 1e6
                max_ms = result.worst / 1e6
                print(f"  Benchmark {idx}: {mean_ms:.4f} ms (min={min_ms:.4f}, max={max_ms:.4f})  {bench.spec}")
            else:
                print(f"  Benchmark {idx}: FAIL (correctness)  {bench.spec}")
                print(f"               {result}")
                exit_code = 1

    return exit_code


def main():
    fd = os.getenv("POPCORN_FD")
    if not fd:
        return run_local()

    if len(sys.argv) < 3:
        return 2

    from utils import set_seed

    mode = sys.argv[1]
    seed = os.getenv("POPCORN_SEED")
    os.unsetenv("POPCORN_SEED")
    seed = int(seed) if seed else None
    set_seed(seed or 42)
    tests = get_test_cases(sys.argv[2], seed)

    with PopcornOutput(int(fd)) as logger:
        import multiprocessing
        mp_context = multiprocessing.get_context('spawn')
        with mp_context.Pool(1) as pool:
            if mode == "test":
                return run_testing(logger, pool, tests)
            if mode == "benchmark":
                return run_benchmarking(logger, pool, tests)

            if mode == "leaderboard":
                # warmup
                run_single_benchmark(pool, tests[0], False, 20)
                logger.log("benchmark-count", len(tests))
                passed = True
                for i in range(len(tests)):
                    result = run_single_benchmark(pool, tests[i], True, 200)
                    logger.log(f"benchmark.{i}.spec", tests[i].spec)
                    if isinstance(result, Stats):
                        for field in dataclasses.fields(Stats):
                            logger.log(f"benchmark.{i}.{field.name}", getattr(result, field.name))
                    else:
                        passed = False
                        logger.log(f"benchmark.{i}.status", "fail")
                        logger.log(f"benchmark.{i}.error", str(result))
                        break

                logger.log("check", "pass" if passed else "fail")
            elif mode == "profile":
                run_profiling(logger, tests)
            else:
                # TODO: Implement script mode
                return 2


if __name__ == "__main__":
    sys.exit(main())
