---
sidebar_position: 2
---

# Creating a CUDA Leaderboard
This section describes how to create CUDA-based leaderboards, which expect CUDA/C++ submissions. 
To create leaderboards on a Discord server, the
Discord bot expects you to have a `Leaderboard Admin` or `Leaderboard Creator` role. These can be
assigned by admins / owners of the server. Nevertheless, this section is also useful for submitters
to understand the details of a leaderboard under the hood.

Like we've mentioned before, each leaderboard specifies a number of GPUs to evaluate on based on the
creator's choosing. You can think of each `(leaderboard, GPU)` pair as *essentially an independent
leaderboard*, as for example, a softmax kernel on an NVIDIA T4 may be very different on an H100. We
give leaderboard creators the option to select which GPUs they care about for their leaderboards --
for example, they may only care about NVIDIA A100 and NVIDIA H100 performance for their leaderboard.

To create a CUDA leaderboard given the correct role permissions, you can run (type it out so it fills in
correctly)
<center>
```
/leaderboard create {leaderboard_name: str} {deadline: str} {reference_code: .cu / .cuh / .cpp file}
```
</center>

After running this, similar to leaderboard submissions, a UI window will pop up asking which GPUs
the leaderboard creator wants to allow submissions on. After selecting the GPUs, the leaderboard
will be created, and users can submit. In the rest of this page, we will explain how to write a
proper `reference_code.cuh` file to create your own leaderboards.

## The Evaluation Harness
When a user submits a CUDA kernel submission to your leaderboard, we use the reference code from
the leaderboard and an evalulation script to check for correctness of the user kernel and measure
the runtime. Unlike in the Python leaderboard, we will compile the evalulation code as the main source
file, and the leaderboard reference / user submission code as header files. We handle re-naming
filenames on our end, so leaderboard creators can use any file extension for their reference
kernels. **It is just important to know that we treat the reference code as a header file
internally.** In all:
* `eval.cu` **(treated as main)**: Run user and reference kernels to check correctness, then measure
  runtime of user kernel if it passes correctness checks.
* `reference_code.cuh`: Define input/output types, data generator, reference kernel, and correctness
  logic to compare user and reference kernel outputs.
* `submission.cuh`: Define user submitted kernel.

The evaluation harness is the same for all CUDA leaderboards, and can be retrieved with
<center>
```
/leaderboard eval-code language:cuda
```
</center>

Let's break down what's going on in this relatively short file:

```cpp title="eval.cu"
#include <chrono>
#include <iostream>

#include "reference.cuh"
#include "train.cuh"

#define WARMUP_RUNS 10
#define TIMED_RUNS 100


float measure_runtime() {
    std::cout << "warming up..." << std::endl;

    for (int i = 0; i < WARMUP_RUNS; i++) {
        auto data = generate_input();
        custom_kernel(data);
    }
    cudaDeviceSynchronize();

    using double_duration = std::chrono::duration<double>;
    double total_duration = 0.0;

    for (int i = 0; i < TIMED_RUNS; i++) {
        auto data = generate_input();

        auto start = std::chrono::high_resolution_clock::now();
        auto submission_output = custom_kernel(data);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        total_duration += std::chrono::duration_cast<double_duration>(end - start).count();

        auto reference_output = ref_kernel(data);
        if (!check_implementation(submission_output, reference_output)) {
            std::cout << "check_implementation failed" << std::endl;
            return 1;
        }

    }


    double average_duration = total_duration / TIMED_RUNS;
    std::cout << "submitted kernel runtime: " << average_duration << " seconds" << std::endl;
    return average_duration;
}

int main() {
    auto data = generate_input();
    auto reference_output = ref_kernel(data);
    auto submission_output = custom_kernel(data);

    if (!check_implementation(submission_output, reference_output)) {
        std::cout << "check_implementation failed" << std::endl;
        return 1;
    }

    float s = measure_runtime();
    if (s < 0) {
        return 1;
    }

    std::cout << "score: " << s << std::endl;

    return 0;
}
```
You'll notice that we include from headers named `reference.cuh` and `train.cuh`. These are the reference
code and submission code respectively, just renamed to a fix module so we can include them. The
general idea is that the evaluation code can treat the leaderboard as a basic abstraction, and only
concern itself with three things:
1. Checking that the reference kernel and user kernel are "equal" (the leaderboard creator defines
what "equal" mean!). This is the `if !check_implementation(...)` line.
2. Warming up the user kernel if it passed correctness checks. This happens in the first part of `measure_runtime()`.
3. Timing the user kernel without including data generation. This happens in the second part of
   `measure_runtime()`.

The abstraction doesn't consider devices either, so the leaderboard creator can choose whether data
starts on the host or on-device -- this kind of flexibility allows leaderboards to evaluate on
whatever the creator is interested in optimizing.

## Reference Code Requirements
The reference code file **must be a `.cu, .cuh, .cpp.`** to create a CUDA leaderboard. `.py`
reference files will create [Python leaderboards](./python-creations). Based on the evaluation harness
above, each reference file **must** have the following function signatures filled out (header guards
are optional but good practice):


```cpp title="reference_template.cuh"
#ifndef __REFERENCE_CUH__
#define __REFERENCE_CUH__

using input_t = ...;
using output_t = ...;

bool check_implementation(output_t custom_output, output_t ref_output) {
    ...
}

input_t generate_input() {
    ...
}

output_t ref_kernel(input_t input) {
    ...
}

#endif
```

We leave it up to the leaderboard creator to fill out these functions and types. This offers the flexibility
of designing a variety of input/output types beyond just lists of Tensors, as well as where the data
is actually placed (e.g. on device or on host). Furthermore, we allow leaderboard creators to define
their own correctness check functions, because some leaderboards may allow for low-precision
submissions through an allowable error such as `rtol` or `atol`.
