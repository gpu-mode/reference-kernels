# Adding Nsight Compute support to a problem

Nsight Compute (NCU) collects NVIDIA GPU hardware counters. To support it, a
problem's evaluator must implement `profile` mode and launch the submitted
kernel inside an NVTX push/pop range named exactly `custom_kernel`. Merely
accepting `profile` and running `torch.profiler` is insufficient.

This guide describes the Python evaluator contract used by KernelBot and
[popcorn-cli](https://github.com/gpu-mode/popcorn-cli). It applies to a single
NVIDIA GPU; AMD profiling requires a different profiler, and the current NCU
runner does not support multi-GPU tasks.

## Choose an evaluator and benchmark shapes

Start with a working example:

- [QR v2 evaluator](../problems/linalg/qr_v2/eval.py): a dedicated NCU-compatible
  profile path, with correctness checking outside the captured range.
- [Shared NVIDIA evaluator](../problems/nvidia/eval.py): separate NCU and
  PyTorch profiler paths. Inspect the evaluator actually selected by your
  `task.yml`; a task-specific evaluator can differ from the shared one.

In `task.yml`, include the evaluator, input generator, submission, and their
imports in `files`, set `config.main` to the evaluator entry point, and provide
at least one representative entry under `benchmarks`. See the
[QR v2 task definition](../problems/linalg/qr_v2/task.yml).

KernelBot profiles `benchmarks`, not `tests`: it launches a separate evaluator
run for each benchmark entry. With the profiling command, `--benchmark-index N`
selects the zero-based `benchmarks[N]`; omitting it profiles every benchmark
entry. The evaluator reads the benchmark specification file supplied by the
runner. Do not hardcode a shape or import a separate shape list for profiling.

Register the problem's leaderboard name, directory, and supported GPU names in
its competition YAML (for example, [linalg.yaml](../problems/linalg.yaml)).
KernelBot syncs that mapping into its leaderboard configuration and rejects
unsupported GPUs.

## Mark the submitted work

Adapt this worker to the task's existing `TestCase`, input cloning, and checking
helpers. `_clone_data` below is the task's cloning helper, not a library API;
use the same input-preservation semantics as your correctness/benchmark paths.
The `check_implementation` signature here follows the QR v2 evaluator.

```python
import torch
from torch.cuda.nvtx import range as nvtx_range

from reference import check_implementation, generate_input


def _run_single_profile_ncu(test):
    from submission import custom_kernel

    data = generate_input(**test.args)
    cloned = _clone_data(data)
    # Finish setup and cloning before entering the captured range.
    torch.cuda.synchronize()

    with nvtx_range("custom_kernel"):
        output = custom_kernel(cloned)
        torch.cuda.synchronize()

    # Keep reference work and correctness checking outside the captured range.
    return check_implementation(data, output)
```

The runner filters with `--nvtx --nvtx-include 'custom_kernel/'`. The trailing
slash selects a push/pop range; the range name in Python has no slash.
`torch.profiler.record_function("custom_kernel")` alone does not satisfy this
contract. A native CUDA evaluator needs the equivalent NVTX push/pop range,
with the NVTX headers and linking appropriate to its build.

Capture one call to the submitted implementation, rather than the benchmark's
repeated timing loop. That call may launch multiple CUDA kernels. NCU performs
its own replay to collect counters. If the implementation requires warmup or
JIT compilation before capture, do it outside the range, restore any mutated
inputs, and synchronize before entering the range.

If the evaluator uses worker processes, define the worker at module scope and
use the existing pool. The runner must use `--target-processes all` to capture
kernels launched in those children; the hosted NCU runner already does this.

## Dispatch and report profile results

Retain the evaluator's normal argument parsing, seeding, worker setup, and
`POPCORN_FD` logging protocol. For an evaluator with the QR-style checker, the
profile dispatcher can be:

```python
def run_profiling(logger, pool, tests):
    logger.log("benchmark-count", len(tests))
    passed = True
    for idx, test in enumerate(tests):
        logger.log(f"benchmark.{idx}.spec", test.spec)
        good, message = pool.apply(_run_single_profile_ncu, (test,))
        logger.log(f"benchmark.{idx}.status", "pass" if good else "fail")
        if not good:
            logger.log(f"benchmark.{idx}.error", message)
            passed = False
    logger.log("check", "pass" if passed else "fail")
    return 0 if passed else 112
```

In the existing `main`, handle `mode == "profile"` by returning
`run_profiling(logger, pool, tests)`. Preserve `sys.exit(main())` so the exit
status reaches the runner. Here `tests` is the evaluator's parsed input list;
for profile mode its contents come from the runner's benchmark file.

Do not invoke `ncu` or write `.ncu-rep` files from the evaluator. The runner
wraps the evaluator in NCU, applies capture options, and packages the reports.

If you keep a PyTorch profiler path, select the NCU worker explicitly:

```python
if os.environ.get("POPCORN_NCU") == "1":
    return pool.apply(_run_single_profile_ncu, (test,))
return pool.apply(_run_single_profile_torch, (test,))
```

This snippet assumes `import os` and both workers exist. Avoid
`bool(os.getenv("POPCORN_NCU", "0"))`: the string `"0"` is truthy. Do not run
`torch.profiler` inside the NCU path; both profilers can compete for profiling
resources. An NCU-only profile path, like QR v2's, can call its worker directly.

## Validate on a real GPU

Use a popcorn-cli build that exposes `submit --profile`; check `popcorn submit
--help` or see the [CLI profiling guide](https://github.com/gpu-mode/popcorn-cli/blob/main/docs/profiling.md).
`--profile` submits through the normal authenticated GPU Mode API. Users do
not need a compute-provider SDK, account, or token. `--profile-brev` explicitly
selects the separate Brev service; there is no automatic provider switch on failure.

Register with Popcorn, then profile a known-correct starter submission:

```bash
popcorn register discord
popcorn submit submission.py --leaderboard YOUR_LEADERBOARD --gpu B200 \
  --profile --benchmark-index 0
```

Use a GPU declared by the problem. Check all of the following before claiming
the new problem supports NCU:

1. The run resolves the intended problem directory and benchmark specification.
2. A nonempty `.ncu-rep` is saved and can be reopened by NCU.
3. `ncu-details.txt` or `ncu-details.csv` names the intended submitted kernel and
   contains actual counter values. A successful process exit or an empty zip is
   not proof of capture.
4. A second, different benchmark index captures the corresponding shape. Check
   the remaining shapes as appropriate for the task; one small shape does not
   establish coverage for all workloads.
5. Normal `test` and `benchmark` modes still work after evaluator changes.

The default capture limit is 10 kernel launches per benchmark. To reach a late
kernel, increase `--ncu-launch-count` or select it explicitly:

```bash
popcorn submit submission.py --leaderboard YOUR_LEADERBOARD --gpu B200 \
  --profile --benchmark-index 0 \
  --ncu-kernel-name 'regex:your_kernel' --ncu-kernel-name-base demangled \
  --ncu-launch-count 1
```

A filter that matches no launched kernels cannot produce a useful profile.
`--set full` requests a broad metric set but does not guarantee that every
metric exists or has data on every GPU. The hosted NCU runner leaves GPU clocks
unchanged, so use the normal benchmark path for latency comparisons.

## Record source provenance

The hosted profiler runs the task/evaluator synced into KernelBot's leaderboard
configuration. Editing a local `eval.py` or setting a client-side repository ref
does not update that configuration. Publish the problem revision and have an
operator sync it through the normal KernelBot problem-update workflow before
validating the hosted profile.

The CLI saves the leaderboard, GPU, selected shapes, capture options, and a
SHA-256 digest of the evaluation configuration in `manifest.json`. This digest
identifies evaluated content, not a repository commit. Record the synced
reference-kernels commit separately in the problem PR or operator validation
record. Brev has its own deployed checkout and must be refreshed separately.

In the problem PR, record the resolved problem directory, reference-kernels
commit, GPU, benchmark indices tested, kernel filter/launch limit, and observed
artifacts. For example, the NCU GPU integration was verified against
`problems/linalg/qr_v2`, benchmark 0, on B200 at reference-kernels
`51e22db671d36c1c76091c43c36a44546ba324a1`. That is evidence for that run, not a
claim that every problem in this repository already supports NCU.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `profile` mode is rejected | Add real profile dispatch in the evaluator selected by `task.yml`. |
| No kernels captured | Check the exact NVTX push/pop range name, matching kernel filter, and child-process capture. |
| Only setup kernels appear | Move generation, cloning, and reference work outside the range; select the intended kernel or expand the launch limit. |
| Profiling initialization fails | Ensure the NCU path does not also start PyTorch's profiler; inspect the NCU version and driver/runtime errors. |
| Wrong shape or missing recent evaluator changes | Inspect `manifest.json` and the operator's synced source revision; local edits are not automatically uploaded. |
| Timeout | Start with one shape and a bounded kernel capture; NCU replay can be much slower than a normal benchmark. |
