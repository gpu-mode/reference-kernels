## Reference Kernels

This repo holds reference kernels for the KernelBot which hosts regular competitions on [discord.gg/gpumode](discord.gg/gpumode).

You can see what's going on [gpumode.com](https://www.gpumode.com/)

## Competition
1. [PMPP practice problems](https://gpu-mode.github.io/discord-cluster-manager/docs/active#practice-round-leaderboard): Starting on Sunday Feb 21, 2025.
2. [AMD $100K kernel competition](problems/amd)

## Making a Leaderboard Submission

Please take a look at `vectoradd_py` to see multiple examples of expected submisisons ranging from PyTorch code to Triton to inline CUDA.


## Contributing New Problems

To add a new problem, create a new folder in the `problems/glory` directory where you need to add the following files:
- `reference.py` - This is the PyTorch reference implementation of the problem.
- `task.yml` - This is the problem specification that will be used to generate test cases for different shapes
- `task.py` - Specifies the schema of the inputs and outputs for the problem




