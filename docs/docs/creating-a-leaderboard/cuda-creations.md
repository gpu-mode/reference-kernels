---
sidebar_position: 2
---

# Creating a CUDA/C++ Leaderboard
This section describes how to create CUDA-based leaderboards, which expect CUDA submissions. **To create leaderboards on a Discord server,
the Discord bot expects you to have a `Leaderboard Admin` or `Leaderboard Creator` role**. These can be
assigned by admins / owners of the server. Nevertheless, this section is also useful for participants
to understand how their submissions are evaluated.

Like we've mentioned before, each leaderboard specifies a number of GPUs to evaluate on based on the
creator's choosing. You can think of each `(task, GPU)` pair as having *essentially its own independent
leaderboard*, as for example, a softmax kernel on an NVIDIA T4 may perform very differently on an NVIDIA H100. We
give leaderboard creators the option to select which GPUs they care about for their leaderboards --
for example, they may only care about NVIDIA A100 and NVIDIA H100 performance for their leaderboard.

To create a leaderboard you can run:
<center>
```
/leaderboard create {leaderboard_name: str} {deadline: str} {task_zip: .zipped folder}
```
</center>

After running this, similar to leaderboard submissions, a UI window will pop up asking which GPUs
the leaderboard creator wants to enable submissions on. In the remaining section, we detail how
the unzipped `task_zip` folder should be structured. [Examples of these folders can be found here.](https://github.com/gpu-mode/discord-cluster-manager/tree/main/examples)

## The `task.yml` specification.
When a user submits a reference kernel, it is launched inside of a leaderboard-specific evaluation harness, and we provide
several [copy-able examples of a leaderboard folder in our GitHub](https://github.com/gpu-mode/discord-cluster-manager/tree/main/examples). 
The relevant files are defined in a `task.yml` -- for example, in the `identity-cuda` leaderboard, the YAML looks as follows:
```yaml title="task.yml"
# What files are involved in leaderboard evaluation
files:
  - {"name": "eval.cu", "source": "eval.cu"}
  - {"name": "task.h", "source": "task.h"}
  - {"name": "utils.h", "source": "utils.h"}
  - {"name": "reference.cuh", "source": "reference.cuh"}
  - {"name": "submission.cu", "source": "@SUBMISSION@"}

# Leaderboard language
lang: "cu"


# Description of leaderboard task
description:
  Identity kernel in CUDA.

# Compilation flag for what to target as main
config:
  # task provided source files to compile
  sources: ["eval.cu", "submission.cu"]

  # additional include directories
  include_dirs: []

# An example to provide to participants for writing a leaderboard submission
templates:
  CUDA: "template.cu"

tests:
  - {"size": 128, "seed": 5236}
  - {"size": 129, "seed": 1001}
  - {"size": 256, "seed": 5531}

benchmarks:
  - {"size": 1024, "seed": 54352}
  - {"size": 4096, "seed": 6256}
  - {"size": 16384, "seed": 6252}
  - {"size": 65536, "seed": 125432}
```

This config file controls all relevant details about how participant will interact with the leaderboard. We will discuss each
parameter in detail. Some of the more simple keys are:
* `lang` controls the language of the leaderboard (`py` or `cu`)
* `config.sources` controls what files are compiled as source files. Most leaderboards will just use `eval.cu` and `submission.cu`.
* `config.include_dirs` specifies extra include directories for say header libraries. By default, this includes `ThunderKittens`.
* `templates` is an optional way to provide users with an example template for a kernel submission.

### Required files in the leaderboard `.zip`
Other than `task.yml`, the `files` key controls the list of files that the evaluation harness expects. The leaderboard
creator has to include all of these files, but we provide examples to make it a lot easier. The
`name` key is how this file is imported locally, and the `source` key is the name of the actual file in the folder.

* `submission.cu`: This is a special key-value pair that denotes the user submitted kernel (it **should not exist** in the `.zip`).
* `task.h` ⭐: Specifies constants and the input / output type (e.g. arguments) that the leaderboard kernel should expect.
* `utils.h`: Some extra utils that can be used for leaderboard logic. We also include a `TestReporter` class that is used for reporting kernel correctness.
* `reference.cuh` ⭐: Leaderboard-specific logic for generating input data, the reference kernel, and correctness
  logic to compare user and reference kernel outputs.
* `eval.cu`: Run user and reference kernels to check correctness, then measure
  runtime of user kernel if it passes correctness checks. Usually does not need to be edited.

In short, most leaderboard
creators will only have to edit `task.h` and `reference.cuh`, but we will go over how to edit these more in detail.

## A simple `task.h` and `reference.cuh` example

To keep this simple, a leaderboard creator really only needs to specify:
1. The input / output types of the desired leaderboard kernel.
2. A generator that generates input data with specific properties.
3. An actual example reference kernel that serves as ground truth.
4. A comparison function to check for correctness of a user submitted kernel against the reference. We allow leaderboard creators full flexibility to specify things like margin of error.

We recommend following our examples for simplicity, but our task definition allows leaderboard creators to fully modify their evaluation harness. In the remaining sections, we will go over how to use our pre-defined examples. 
In all of our examples, the `task.h` file handles (1) and part of (2), while the `reference.cuh` file handles (2,3,4). Below, we provide the `task.py` for the `identity-cuda` leaderboard.

```cpp title="task.h"
#ifndef __POPCORN_TASK_H__
#define __POPCORN_TASK_H__

#include <vector>
#include <array>

using input_t = std::vector<float>;
using output_t = input_t;

constexpr std::array<const char*, 2> ArgumentNames = {"seed", "size"};

#endif
```

The example above specifies aliases for the input (`input_t`) and output (`output_t`) types of the kernel task. It also specifies
an array of strings called `ArgumentNames`, which specifies **what arguments are passed into the input data generator** at runtime. We distinguish
between `tests` cases and `benchmarks` cases, the former being the actual leaderboard cases and the latter being for users to 
debug their code. Using this `TestSpec` specification, we provide test cases to the `task.yml` and fill in the arguments, as shown below:


```yaml title="task.yml"
...
tests:
  - {"size": 128, "seed": 5236}
  - {"size": 129, "seed": 1001}
  - {"size": 256, "seed": 5531}

benchmarks:
  - {"size": 1024, "seed": 54352}
  - {"size": 4096, "seed": 6256}
  - {"size": 16384, "seed": 6252}
  - {"size": 65536, "seed": 125432}
```

Finally, we fill in details for the input data generator, reference kernel, and correctness checker for `identity-cuda` below:

```cpp title="reference.cuh
#ifndef __REFERENCE_CUH__
#define __REFERENCE_CUH__

#include <tuple>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <array>
#include <random>
#include <iostream>

#include "task.h"

// Input data generator. Arguments must match ArgumentNames in task.h
static input_t generate_input(int seed, int size) {
  std::mt19937 rng(seed);
  input_t data;
  std::uniform_real_distribution<float> dist(0, 1);

  data.resize(size);
  for (int j = 0; j < size; ++j) {
    data[j] = dist(rng);
  }

  return data;
}

// Reference kernel. Must take `input_t` and produce `output_t`
static output_t ref_kernel(input_t data) {
  return (output_t) data;
}

static void check_implementation(TestReporter& reporter, input_t data, output_t user_out, float epsilon = 1e-5) {
  output_t ref = ref_kernel(data);

  if(user_out.size() != ref.size()) {
      if(!reporter.check_equal("size mismatch", user_out.size(), ref.size())) return;
  }

  for (int j = 0; j < ref.size(); ++j) {
    if (std::fabs(ref[j] - user_out[j]) > epsilon) {
        reporter.fail() << "error at " << j << ": " << ref[j] << " "  << std::to_string(user_out[j]);
        return;
    }
  }

  reporter.pass();
}

#endif
```

As mentioned earlier, based on `task.yml` and `task.h`, each test case will pass a specified set
of arguments to `generate_input(...)` to produce the input data for that task case. We recommend specifying 
a seed argument to properly randomizing inputs in a reproducible manner. Furthermore, `check_implementation` uses a 
a `TestReport` object defined in `utils.h` to give leaderboard creators the flexibility to provide error messages to participants to help debug
and easily specify whether test cases pass/fail. 

**Remark.** Leaderboard creators have the flexibility to edit the logic in `eval.cu`, which uses all of these functions
to evaluate and measure the user specified kernels. The examples above assume the use of our `eval.cu` implementation, but
this can be modified if desired.

## Deleting a Leaderboard
If you have sufficient permissions on the server, you can also delete leaderboards with:

<center>
```
/leaderboard delete {leaderboard_name: str}
```
</center>

This command will display a UI window with a list of available leaderboards. Select the leaderboard you want to delete from the list. Once confirmed, the leaderboard and all associated submissions will be permanently removed. Please use this command with caution, as it will also delete the leaderboard history as well.

## Existing Leaderboard Examples
We try to provide examples of leaderboards that can be quickly copied and modified for other references [here](https://github.com/gpu-mode/discord-cluster-manager/tree/main/examples). 
Most leaderboards should be able to just modify these files.
