---
sidebar_position: 1
---

# Creating a Python Leaderboard
This section describes how to create Python-based leaderboards, which expect Python submissions
(they can still [inline compile
CUDA](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load_inline) code though). To create leaderboards on a Discord server, the
Discord bot expects you to have a `Leaderboard Admin` or `Leaderboard Creator` role. These can be
assigned by admins / owners of the server. Nevertheless, this section is also useful for submitters
to understand the details of a leaderboard under the hood.

Like we've mentioned before, each leaderboard specifies a number of GPUs to evaluate on based on the
creator's choosing. You can think of each `(leaderboard, GPU)` pair as *essentially an independent
leaderboard*, as for example, a softmax kernel on an NVIDIA T4 may be very different on an H100. We
give leaderboard creators the option to select which GPUs they care about for their leaderboards --
for example, they may only care about NVIDIA A100 and NVIDIA H100 performance for their leaderboard.

To create a Python leaderboard given the correct role permissions, you can run (type it out so it fills in
correctly)
<center>
```
/leaderboard create {leaderboard_name: str} {deadline: str} {reference_code: .py file}
```
</center>

After running this, similar to leaderboard submissions, a UI window will pop up asking which GPUs
the leaderboard creator wants to allow submissions on. After selecting the GPUs, the leaderboard
will be created, and users can submit. In the rest of this page, we will explain how to write a
proper `reference_code.py` file to create your own leaderboards.

## The Evaluation Harness
When a user submits a Python kernel submission to your leaderboard, we use the reference code from
the leaderboard and an evalulation script to check for correctness of the user kernel and measure
the runtime. In all:
* `eval.py` **(treated as main)**: Run user and reference kernels to check correctness, then measure
  runtime of user kernel if it passes correctness checks.
* `reference_code.py`: Define input/output types, data generator, reference kernel, and correctness
  logic to compare user and reference kernel outputs.
* `submission.py`: Define user submitted kernel.

The evaluation harness is the same for all Python leaderboards, and can be retrieved with
<center>
```
/leaderboard eval-code language:python
```
</center>

Let's break down what's going on in this relatively short file:

```python title="eval.py"

import torch
import time
from reference import ref_kernel, generate_input, check_implementation
from train import custom_kernel


def correctness() -> bool:
    for _ in range(10):  # check multiple times
        inputs = generate_input()
        custom_output = custom_kernel(inputs)
        ref_output = ref_kernel(inputs)

        # User leaderboard-defined "equality" to check correctness
        if not check_implementation(custom_output, ref_output):
            return False
    return True


def metric():
    warmup_runs = 10
    timed_runs = 100

    # Warmup Code
    for _ in range(warmup_runs):
        inputs = generate_input()
        _ = custom_kernel(inputs)
    torch.cuda.synchronize()

    # Timing Code
    inputs = generate_input()
    start_time = time.time()
    for _ in range(timed_runs):
        _ = custom_kernel(inputs)
    torch.cuda.synchronize()
    end_time = time.time()

    custom_duration = (end_time - start_time) / timed_runs
    print(f'Submitted kernel runtime: {custom_duration:.4f} seconds')

    return custom_duration

def main():
    assert (correctness())

    # Warmup + Profile runtime
    s = metric()
    print(f'score:{s}')

if __name__ == '__main__':
    main()
```
You'll notice that we import from a module named `reference` and `train`. These are the reference
code and submission code respectively, just renamed to a fix module so we can import them. The
general idea is that the evaluation code can treat the leaderboard as a basic abstraction, and only
concern itself with three things:
1. Checking that the reference kernel and user kernel are "equal" (the leaderboard creator defines
what "equal" mean!). This is the `assert(correctness())` line.
2. Warming up the user kernel if it passed correctness checks. This happens in the first part of `metric()`.
3. Timing the user kernel without including data generation. This happens in the second part of
   `metric()`.

The abstraction doesn't consider devices either, so the leaderboard creator can choose whether data
starts on the host or on-device -- this kind of flexibility allows leaderboards to evaluate on
whatever the creator is interested in optimizing.

## Reference Code Requirements
The reference code file **must be a `.py`** to create a Python leaderboard. `.cu, .cuh, .cpp`
reference files will create [CUDA leaderboards](./cuda-creations). Based on the evaluation harness
above, each reference file **must** have the following function signatures filled out:


```python title="reference_template.py"

def check_implementation(
        user_output: OutputType,
        reference_output: OutputType,
    ) -> bool:
    ...

def generate_input() -> InputType:
    ...

def ref_kernel(data: InputType) -> OutputType:
    ...
```

We leave it up to the leaderboard creator to fill out these functions and types. This offers the flexibility
of designing a variety of input/output types beyond just lists of Tensors, as well as where the data
is actually placed (e.g. on device or on host). Furthermore, we allow leaderboard creators to define
their own correctness check functions, because some leaderboards may allow for low-precision
submissions through an allowable error such as `rtol` or `atol`.
