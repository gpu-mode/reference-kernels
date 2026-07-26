"""Application-level validation for ranked Cholesky submissions.

KernelBot runs this entrypoint in an isolated Modal process. The problem-owned
configuration is supplied in ``KERNELBOT_VALIDATION_CONFIG`` and the submitted
entrypoint is available as ``submission.custom_kernel``.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import statistics
import time
from typing import Callable

import torch
import torch.nn.functional as F


Tensor = torch.Tensor

SHAPES = [
    (4096, 32),
    (1024, 64),
    (256, 128),
    (64, 256),
    (16, 512),
    (4, 1024),
    (2, 2048),
    (1, 4096),
]
STEPS = 12
SAMPLES = 48
SEED = 20260726
LEARNING_RATE = 0.15
RIDGE = 1.0e-5
MAX_RELATIVE_RESIDUAL = 5.0e-4
BENCHMARK_WARMUP = 3
BENCHMARK_REPEATS = 5


def _fwht(values: Tensor) -> Tensor:
    n = values.shape[-1]
    if n <= 0 or n & (n - 1):
        raise ValueError(f"validation width must be a power of two, got {n}")
    out = values
    block = 1
    while block < n:
        grouped = out.reshape(*out.shape[:-1], -1, 2, block)
        left = grouped[..., 0, :]
        right = grouped[..., 1, :]
        out = torch.cat((left + right, left - right), dim=-1).reshape_as(out)
        block *= 2
    return out * (1.0 / math.sqrt(float(n)))


def _make_problem(
    *,
    tasks: int,
    samples: int,
    n: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    scales = torch.logspace(0.0, -2.5, n, device="cuda", dtype=torch.float32)
    signs = torch.randint(
        0,
        2,
        (n,),
        device="cuda",
        dtype=torch.int64,
        generator=generator,
    ).to(torch.float32)
    signs.mul_(2.0).sub_(1.0)
    permutation = torch.randperm(n, device="cuda", generator=generator)
    z = torch.randn(
        (tasks, samples, n),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    x = _fwht((z * scales).index_select(-1, permutation) * signs).contiguous()
    target_weights = torch.randn(
        (tasks, n),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    target_weights.mul_(6.0 / math.sqrt(float(n)))
    targets = torch.sigmoid(
        torch.einsum("tmn,tn->tm", x, target_weights)
    ).contiguous()
    return x, targets


def _damping(step: int, steps: int) -> float:
    fraction = step / max(steps, 1)
    if fraction < 0.15:
        return 3.0e-2
    if fraction < 0.30:
        return 3.0e-3
    if fraction < 0.50:
        return 1.0e-3
    if fraction < 0.70:
        return 3.0e-4
    if fraction < 0.85:
        return 1.0e-4
    return 1.0e-5


def _validate_factor(
    matrix: Tensor,
    factor: Tensor,
    *,
    max_relative_residual: float,
) -> float:
    if type(factor) is not torch.Tensor:
        raise TypeError(f"custom_kernel returned {type(factor)!r}")
    if factor.shape != matrix.shape:
        raise ValueError(
            f"factor shape {tuple(factor.shape)} does not match {tuple(matrix.shape)}"
        )
    if factor.dtype != torch.float32 or not factor.is_cuda:
        raise ValueError("factor must be a float32 CUDA tensor")
    if not torch.isfinite(factor).all().item():
        raise FloatingPointError("factor contains NaN or Inf")

    diagonal_min = float(
        torch.diagonal(factor, dim1=-2, dim2=-1).amin().item()
    )
    if diagonal_min <= 0.0:
        raise FloatingPointError(f"factor diagonal is not positive: {diagonal_min}")
    upper_abs = float(torch.triu(factor, diagonal=1).abs().amax().item())
    if upper_abs > 1.0e-5:
        raise FloatingPointError(f"factor has upper-triangle leakage: {upper_abs}")

    reconstruction = torch.matmul(factor, factor.transpose(-1, -2))
    residual = torch.linalg.matrix_norm(
        reconstruction - matrix,
        ord="fro",
        dim=(-2, -1),
    ) / torch.linalg.matrix_norm(
        matrix,
        ord="fro",
        dim=(-2, -1),
    ).clamp_min(1.0e-30)
    worst_residual = float(residual.amax().item())
    if worst_residual > max_relative_residual:
        raise FloatingPointError(
            "factor reconstruction residual exceeds gate: "
            f"{worst_residual} > {max_relative_residual}"
        )
    return worst_residual


def _train(
    factor_fn: Callable[[Tensor], Tensor],
    *,
    batch: int,
    n: int,
    steps: int,
    samples: int,
    seed: int,
    learning_rate: float,
    ridge: float,
    max_relative_residual: float,
) -> dict[str, float | int]:
    x, targets = _make_problem(
        tasks=batch,
        samples=samples,
        n=n,
        seed=seed,
    )
    weights = torch.zeros((batch, n), device="cuda", dtype=torch.float32)
    eye = torch.eye(n, device="cuda", dtype=torch.float32).expand(batch, -1, -1)
    initial_loss = None
    final_loss = None
    worst_residual = 0.0

    for step in range(steps):
        logits = torch.einsum("tmn,tn->tm", x, weights)
        probabilities = torch.sigmoid(logits)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("loss became non-finite")
        error = probabilities - targets
        gradient = torch.einsum("tmn,tm->tn", x, error) / float(samples)
        gradient = gradient + ridge * weights
        fisher_weight = (
            probabilities * (1.0 - probabilities)
        ).clamp_min(1.0e-6)
        weighted_x = x * fisher_weight.sqrt().unsqueeze(-1)
        fisher = torch.matmul(
            weighted_x.transpose(-1, -2),
            weighted_x,
        ) / float(samples)
        fisher = (
            fisher + (_damping(step, steps) + ridge) * eye
        ).contiguous()

        factor = factor_fn(fisher.clone())
        factor = factor.clone()
        torch.cuda.synchronize()
        if step in (0, steps - 1):
            residual = _validate_factor(
                fisher,
                factor,
                max_relative_residual=max_relative_residual,
            )
            worst_residual = max(worst_residual, residual)
        elif not torch.isfinite(factor).all().item():
            raise FloatingPointError("factor contains NaN or Inf")

        direction = torch.cholesky_solve(
            gradient.unsqueeze(-1),
            factor,
        ).squeeze(-1)
        if not torch.isfinite(direction).all().item():
            raise FloatingPointError("preconditioned update contains NaN or Inf")
        weights.add_(direction, alpha=-learning_rate)

        loss_value = float(loss.item())
        initial_loss = loss_value if initial_loss is None else initial_loss
        final_loss = loss_value

    return {
        "steps_completed": steps,
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "worst_checked_factor_residual": worst_residual,
    }


def _initial_fisher(
    *,
    batch: int,
    n: int,
    samples: int,
    seed: int,
    ridge: float,
) -> Tensor:
    x, _ = _make_problem(tasks=batch, samples=samples, n=n, seed=seed)
    weighted_x = x * 0.5
    fisher = torch.matmul(
        weighted_x.transpose(-1, -2),
        weighted_x,
    ) / float(samples)
    eye = torch.eye(n, device="cuda", dtype=torch.float32).expand(batch, -1, -1)
    return (fisher + (3.0e-2 + ridge) * eye).contiguous()


def _benchmark_wall(
    factor_fn: Callable[[Tensor], Tensor],
    inputs: list[Tensor],
    *,
    warmup: int,
    repeats: int,
) -> float:
    for index in range(warmup):
        factor_fn(inputs[index % len(inputs)].clone()).clone()
    torch.cuda.synchronize()
    samples = []
    for index in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        factor_fn(inputs[index % len(inputs)].clone()).clone()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1.0e6)
    return statistics.median(samples)


def _run() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Cholesky validation")
    config = json.loads(os.environ["KERNELBOT_VALIDATION_CONFIG"])
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    def baseline_fn(matrix: Tensor) -> Tensor:
        return torch.linalg.cholesky_ex(
            matrix,
            check_errors=False,
        ).L

    submission = importlib.import_module("submission")
    factor_fn = getattr(submission, "custom_kernel", None)
    if not callable(factor_fn):
        raise AttributeError("submission must define callable custom_kernel")

    results: list[dict[str, object]] = []
    for index, (batch, n) in enumerate(SHAPES):
        item: dict[str, object] = {
            "batch": batch,
            "n": n,
            "steps": STEPS,
            "passed": False,
        }
        try:
            matrix_inputs = [
                _initial_fisher(
                    batch=batch,
                    n=n,
                    samples=SAMPLES,
                    seed=SEED + index * 100 + ring,
                    ridge=RIDGE,
                )
                for ring in range(2)
            ]
            baseline_training = _train(
                baseline_fn,
                batch=batch,
                n=n,
                steps=STEPS,
                samples=SAMPLES,
                seed=SEED + index,
                learning_rate=LEARNING_RATE,
                ridge=RIDGE,
                max_relative_residual=MAX_RELATIVE_RESIDUAL,
            )
            factor = factor_fn(matrix_inputs[0].clone()).clone()
            torch.cuda.synchronize()
            direct_residual = _validate_factor(
                matrix_inputs[0],
                factor,
                max_relative_residual=MAX_RELATIVE_RESIDUAL,
            )
            baseline_wall_us = _benchmark_wall(
                baseline_fn,
                matrix_inputs,
                warmup=BENCHMARK_WARMUP,
                repeats=BENCHMARK_REPEATS,
            )
            candidate_wall_us = _benchmark_wall(
                factor_fn,
                matrix_inputs,
                warmup=BENCHMARK_WARMUP,
                repeats=BENCHMARK_REPEATS,
            )
            candidate_training = _train(
                factor_fn,
                batch=batch,
                n=n,
                steps=STEPS,
                samples=SAMPLES,
                seed=SEED + index,
                learning_rate=LEARNING_RATE,
                ridge=RIDGE,
                max_relative_residual=MAX_RELATIVE_RESIDUAL,
            )
            speedup = baseline_wall_us / candidate_wall_us
            baseline_improvement = (
                baseline_training["initial_loss"]
                - baseline_training["final_loss"]
            )
            loss_threshold = (
                baseline_training["final_loss"] + 0.05 * baseline_improvement
            )
            converged = candidate_training["final_loss"] <= loss_threshold
            fast_enough = speedup >= 1.0
            passed = converged and fast_enough
            item.update(
                {
                    "status": "completed",
                    "passed": passed,
                    "numerically_stable": True,
                    "converged": converged,
                    "sync_wall_speedup": speedup,
                    "factor_residual": max(
                        direct_residual,
                        float(
                            candidate_training["worst_checked_factor_residual"]
                        ),
                    ),
                    "baseline_final_loss": baseline_training["final_loss"],
                    "candidate_final_loss": candidate_training["final_loss"],
                }
            )
            if not passed:
                failed_gates = []
                if not converged:
                    failed_gates.append("convergence")
                if not fast_enough:
                    failed_gates.append("speed")
                item["failure_reason"] = ", ".join(failed_gates)
        except Exception as exc:
            item.update(
                {
                    "status": "failed",
                    "passed": False,
                    "numerically_stable": False,
                    "converged": False,
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
        results.append(item)

    passed_shapes = sum(bool(item["passed"]) for item in results)
    speedups = [
        float(item["sync_wall_speedup"])
        for item in results
        if item.get("sync_wall_speedup") is not None
    ]
    geomean_speedup = (
        math.exp(sum(math.log(value) for value in speedups) / len(speedups))
        if speedups
        else None
    )
    return {
        "schema_version": "kernelbot-application-validation-v1",
        "contract_version": config["version"],
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "passed_shapes": passed_shapes,
        "total_shapes": len(SHAPES),
        "fully_validated": passed_shapes == len(results),
        "geomean_sync_wall_speedup": geomean_speedup,
        "results": results,
    }


def main() -> int:
    try:
        result = _run()
    except Exception as exc:
        result = {
            "schema_version": "kernelbot-application-validation-v1",
            "status": "failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "passed_shapes": 0,
            "total_shapes": 0,
            "fully_validated": False,
            "results": [],
        }
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result.get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
