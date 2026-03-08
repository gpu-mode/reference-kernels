# Evaluator Security: In-Process Trust Boundary Issue

## Summary

Several benchmark evaluator families in this repository share the same trust-boundary problem:

1. trusted evaluator code imports mutable live objects from `reference.py` and `utils.py`
2. trusted evaluator then imports untrusted `submission.py` into the same Python interpreter
3. the submission can mutate trusted function objects or timing helpers
4. the evaluator continues to trust those mutated objects for correctness and/or benchmarking

This document summarizes the issue, records what has been directly verified, and proposes a remediation plan.

It intentionally avoids including exploit payloads.

## Affected Evaluator Pattern

The common pattern looks like this:

- import `check_implementation` and `generate_input` from `reference`
- in the worker process, import `custom_kernel` from `submission`
- benchmark by calling:
  - `generate_input(...)`
  - `custom_kernel(...)`
  - `check_implementation(...)`
  - Python-visible timing helpers such as `torch.cuda.Event`

Because all of this happens inside the same interpreter as untrusted code, `submission.py` can interfere with the evaluator’s own trusted logic.

Representative evaluator families using this pattern:

- `problems/amd_202602/eval.py`
- `problems/pmpp_v2/eval.py`
- `problems/nvidia/eval.py`
- `problems/amd/eval.py`
- `problems/amd_distributed/eval.py`
- `problems/helion/eval.py`
- `problems/bioml/trimul/eval.py`
- `problems/pmpp/eval.py`

## What Was Directly Verified

The following claims were directly verified on the live service during March 2026 using evaluator-bypass submissions:

### Verified leaderboard-mode bypasses

- `amd-mixed-mla` (`765`, `MI355X`)
- `amd-moe-mxfp4` (`764`, `MI355X`)
- `amd-mxfp4-mm` (`763`, `MI355X`)
- `matmul_v2` (`540`, `A100`)
- `grayscale_v2` (`538`, `A100`)

### Verified test-mode bypass

- `vectoradd_v2` (`543`, `A100`)

These confirmations indicate that the issue is architectural, not task-specific.

## Why Some Existing Public Scores Are Concerning

Some public leaderboard entries are hard to reconcile with honest evaluation.

One clear example is `matmul_v2`, which includes a benchmark with:

- `m = 4096`
- `n = 5120`
- `k = 4096`

That workload is approximately:

- `2 * m * n * k = 171,798,691,840` floating-point operations

Public entries of `0.001 µs` on this board imply roughly:

- `1.7179869184e20 FLOP/s`
- about `171.8 exaFLOP/s`

for a single device, which is not plausible for the hardware involved.

This does not prove the exact mechanism used by other users, but it is strong evidence that the current evaluator outputs can be invalid.

## Root Cause

The evaluator currently treats the Python interpreter as both:

- the trusted benchmark harness
- the execution environment for untrusted submissions

That is the core mistake.

Even if correctness and timing code are logically separate at the source level, they are still mutable from the submission if they live in the same process and use shared Python objects.

## Recommended Remediation

### Immediate actions

1. Freeze or invalidate suspect leaderboard entries on affected evaluator families.
2. Pause new ranked submissions on affected evaluators until the trust boundary is fixed.
3. Re-run rankings after patching with clean submissions only.

### Short-term mitigation

1. Move correctness checking into a trusted process that never imports `submission.py`.
2. Move benchmark timing into a trusted wrapper process or external profiler-controlled layer.
3. Treat `nan`-based “maximum error” reports as hard failures, not passing messages.
4. Add evaluator self-integrity checks so mutation of trusted function objects causes immediate failure.

### Long-term fix

Define a strict execution model:

- untrusted submission process
- trusted input-generation process
- trusted reference/correctness process
- trusted timing/benchmark process

The submission should never share mutable Python state with the reference implementation or benchmark harness.

## Suggested Design

### Option A: Split-process evaluation

For each run:

1. trusted process generates input
2. trusted process launches isolated submission worker
3. submission worker returns output (or serialized handle/result)
4. trusted process validates correctness using a clean reference environment
5. trusted process measures timing outside the submission interpreter

This is the most robust fix.

### Option B: Transitional hardening

As an interim step:

1. snapshot trusted evaluator symbols before importing or calling untrusted submission code
2. verify those symbols after each submission call
3. fail closed on any mutation

This is not as strong as process isolation, but it would block the straightforward monkeypatch vector.

## What This Document Does Not Do

- It does not include exploit code.
- It does not attribute methods to third-party users.
- It does not claim every leaderboard has been live-tested.

It is a repository-level security note intended to support remediation.
