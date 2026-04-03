import dataclasses
import math
import os
import re
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from reference import (
    ATOL,
    DTYPE,
    RTOL,
    generate_inputs,
    reference_backward,
    reference_forward,
)


# Tri's original eval parameters
B = 4_096
WARMUP_ITERS = 20
BENCH_ITERS = 100


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
    fwd_bw: float
    bwd_bw: float
    combined_bw: float


def get_test_cases(file_name: str) -> list[TestCase]:
    try:
        content = Path(file_name).read_text()
    except Exception as exc:
        print(f"Could not open test file `{file_name}`: {exc}", file=sys.stderr)
        sys.exit(113)

    tests = []
    lines = content.splitlines()
    match = r"\s*([a-zA-Z_]+):\s*([a-zA-Z_]+|[+-]?[0-9]+)\s*"
    for line in lines:
        if not line.strip():
            continue
        parts = line.split(";")
        case = {}
        for part in parts:
            matched = re.match(match, part)
            if not re.fullmatch(match, part):
                print(f"invalid test case: '{line}': '{part}'", file=sys.stderr)
                sys.exit(113)
            key = matched[1]
            value = matched[2]
            try:
                value = int(value)
            except ValueError:
                pass
            case[key] = value
        tests.append(TestCase(spec=line, args=case))
    return tests


def load_submission():
    import submission

    for fn_name in ("cross_entropy_forward", "cross_entropy_backward"):
        if not hasattr(submission, fn_name):
            raise AttributeError(f"Submission is missing function '{fn_name}'.")
    return submission


def check_correctness(mod, vocab_size):
    logits, targets, grad_output = generate_inputs(B, vocab_size)

    ref_loss = reference_forward(logits, targets)
    sub_loss = mod.cross_entropy_forward(logits, targets)

    assert sub_loss.shape == ref_loss.shape, (
        f"Forward shape mismatch: expected {ref_loss.shape}, got {sub_loss.shape}"
    )
    assert sub_loss.dtype == torch.float32, (
        f"Forward dtype mismatch: expected float32, got {sub_loss.dtype}"
    )

    fwd_close = torch.allclose(sub_loss, ref_loss, atol=ATOL, rtol=RTOL)
    max_fwd_err = (sub_loss - ref_loss).abs().max().item()

    ref_grad = reference_backward(logits, targets, grad_output)
    sub_grad = mod.cross_entropy_backward(logits, targets, grad_output)

    assert sub_grad.shape == ref_grad.shape, (
        f"Backward shape mismatch: expected {ref_grad.shape}, got {sub_grad.shape}"
    )
    assert sub_grad.dtype == DTYPE, (
        f"Backward dtype mismatch: expected {DTYPE}, got {sub_grad.dtype}"
    )

    bwd_close = torch.allclose(sub_grad, ref_grad, atol=ATOL, rtol=RTOL)
    max_bwd_err = (sub_grad.float() - ref_grad.float()).abs().max().item()

    return fwd_close, bwd_close, max_fwd_err, max_bwd_err


def benchmark_one(mod, vocab_size):
    logits, targets, grad_output = generate_inputs(B, vocab_size, seed=123)

    for _ in range(WARMUP_ITERS):
        mod.cross_entropy_forward(logits, targets)
        mod.cross_entropy_backward(logits, targets, grad_output)
    torch.cuda.synchronize()

    fwd_times = []
    for _ in range(BENCH_ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        mod.cross_entropy_forward(logits, targets)
        end.record()
        torch.cuda.synchronize()
        fwd_times.append(start.elapsed_time(end))

    bwd_times = []
    for _ in range(BENCH_ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        mod.cross_entropy_backward(logits, targets, grad_output)
        end.record()
        torch.cuda.synchronize()
        bwd_times.append(start.elapsed_time(end))

    combined_times = []
    for _ in range(BENCH_ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        mod.cross_entropy_forward(logits, targets)
        mod.cross_entropy_backward(logits, targets, grad_output)
        end.record()
        torch.cuda.synchronize()
        combined_times.append(start.elapsed_time(end))

    fwd_ms = statistics.median(fwd_times)
    bwd_ms = statistics.median(bwd_times)
    combined_ms = statistics.median(combined_times)

    fwd_bytes = 2 * B * vocab_size + 12 * B
    bwd_bytes = 4 * B * vocab_size + 12 * B
    total_bytes = fwd_bytes + bwd_bytes

    fwd_bw = fwd_bytes / (fwd_ms * 1e-3) / 1e9
    bwd_bw = bwd_bytes / (bwd_ms * 1e-3) / 1e9
    combined_bw = total_bytes / (combined_ms * 1e-3) / 1e9

    # Keep KernelBot scoring on the exact metric Tri reports: median combined ms.
    return Stats(
        runs=BENCH_ITERS,
        mean=combined_ms * 1e6,
        std=statistics.pstdev(combined_times) * 1e6,
        err=(statistics.pstdev(combined_times) / math.sqrt(len(combined_times))) * 1e6,
        best=min(combined_times) * 1e6,
        worst=max(combined_times) * 1e6,
        fwd_bw=fwd_bw,
        bwd_bw=bwd_bw,
        combined_bw=combined_bw,
    )


def run_testing(logger: PopcornOutput, tests: list[TestCase]) -> int:
    try:
        mod = load_submission()
    except Exception as exc:
        logger.log("check", "fail")
        logger.log("error", str(exc))
        return 112

    passed = True
    logger.log("test-count", len(tests))
    for idx, test in enumerate(tests):
        vocab_size = int(test.args["vocab_size"])
        logger.log(f"test.{idx}.spec", test.spec)
        try:
            fwd_ok, bwd_ok, fwd_err, bwd_err = check_correctness(mod, vocab_size)
            if fwd_ok and bwd_ok:
                logger.log(f"test.{idx}.status", "pass")
                logger.log(
                    f"test.{idx}.message",
                    f"forward max err={fwd_err:.3e}, backward max err={bwd_err:.3e}",
                )
            else:
                logger.log(f"test.{idx}.status", "fail")
                logger.log(
                    f"test.{idx}.error",
                    f"forward max err={fwd_err:.3e} {'OK' if fwd_ok else 'FAIL'}; "
                    f"backward max err={bwd_err:.3e} {'OK' if bwd_ok else 'FAIL'}",
                )
                passed = False
        except Exception as exc:
            logger.log(f"test.{idx}.status", "fail")
            logger.log(f"test.{idx}.error", str(exc))
            passed = False

    logger.log("check", "pass" if passed else "fail")
    return 0 if passed else 112


def run_benchmarking(logger: PopcornOutput, tests: list[TestCase]) -> int:
    try:
        mod = load_submission()
    except Exception as exc:
        logger.log("check", "fail")
        logger.log("error", str(exc))
        return 112

    baseline_mod = type(sys)("baseline")
    baseline_mod.cross_entropy_forward = (
        lambda logits, targets: F.cross_entropy(logits.float(), targets, reduction="none")
    )

    def baseline_bwd(logits, targets, grad_output):
        probs = torch.softmax(logits.float(), dim=-1)
        probs[torch.arange(logits.shape[0], device=logits.device), targets] -= 1.0
        return (probs * grad_output.unsqueeze(1)).to(logits.dtype)

    baseline_mod.cross_entropy_backward = baseline_bwd

    passed = True
    logger.log("benchmark-count", len(tests))
    for idx, test in enumerate(tests):
        vocab_size = int(test.args["vocab_size"])
        logger.log(f"benchmark.{idx}.spec", test.spec)
        try:
            baseline = benchmark_one(baseline_mod, vocab_size)
            result = benchmark_one(mod, vocab_size)
            speedup = baseline.mean / result.mean
        except Exception as exc:
            logger.log(f"benchmark.{idx}.status", "fail")
            logger.log(f"benchmark.{idx}.error", str(exc))
            passed = False
            continue

        logger.log(f"benchmark.{idx}.runs", result.runs)
        logger.log(f"benchmark.{idx}.mean", result.mean)
        logger.log(f"benchmark.{idx}.std", result.std)
        logger.log(f"benchmark.{idx}.err", result.err)
        logger.log(f"benchmark.{idx}.best", result.best)
        logger.log(f"benchmark.{idx}.worst", result.worst)
        logger.log(f"benchmark.{idx}.fwd_bw", result.fwd_bw)
        logger.log(f"benchmark.{idx}.bwd_bw", result.bwd_bw)
        logger.log(f"benchmark.{idx}.combined_bw", result.combined_bw)
        logger.log(f"benchmark.{idx}.speedup", speedup)
        logger.log(
            f"benchmark.{idx}.message",
            (
                f"fwd+bwd={result.mean / 1e6:.3f} ms, "
                f"fwd_bw={result.fwd_bw:.1f} GB/s, "
                f"bwd_bw={result.bwd_bw:.1f} GB/s, "
                f"combined_bw={result.combined_bw:.1f} GB/s, "
                f"speedup={speedup:.2f}x"
            ),
        )

    logger.log("check", "pass" if passed else "fail")
    return 0 if passed else 112


def main():
    fd = os.getenv("POPCORN_FD")
    if not fd:
        return 111

    if len(sys.argv) < 3:
        return 2

    if not torch.cuda.is_available():
        with PopcornOutput(int(fd)) as logger:
            logger.log("check", "fail")
            logger.log("error", "No CUDA GPU available. This script requires a GPU.")
        return 112

    mode = sys.argv[1]
    tests = get_test_cases(sys.argv[2])

    with PopcornOutput(int(fd)) as logger:
        if mode == "test":
            return run_testing(logger, tests)
        if mode in {"benchmark", "leaderboard"}:
            return run_benchmarking(logger, tests)

        logger.log("check", "fail")
        logger.log("error", f"Unsupported mode: {mode}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
