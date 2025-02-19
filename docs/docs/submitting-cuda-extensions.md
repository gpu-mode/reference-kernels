---
sidebar_position: 4
---

# Using CUDA Extensions

To use CUDA libraries in submissions (e.g. CUTLASS, ThunderKittens), we explicitly setup and compile them in our evaluation
environment. In this brief section, we go over how to use these libraries.

### CUTLASS in CUDA Submissions
TBD

### ThunderKittens in CUDA Submissions
By default we compile CUDA leaderboards with the ThunderKittens header library (e.g. `ThunderKittens/include`), and users
can use it as if it were available locally:
```cpp title="submission.cu"
#include "kittens.cuh"
using namespace kittens;
...
```

### Triton in Python Submissions
```cpp title="submission.py"
import triton
...
