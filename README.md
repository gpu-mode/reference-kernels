## Reference Kernels

This repo holds reference kernels for the KernelBot which hosts regular competitions on discord.gg/gpumode

## Competition
1. PMPP practice problems: Starting on Sunday Feb 21, 2025
2. LLM competition: Coming soon!

## Making a submission

Please take a look at `vectoradd_py` to see multiple examples of expected submisisons ranging from PyTorch code to Triton to inline CUDA.


## Contributing New Problems

To add a new problem, create a new folder in the `problems/glory` directory where you need to add the following files:
- `reference.py` - This is the PyTorch reference implementation of the problem.
- `task.yml` - This is the problem specification that will be used to generate test cases for different shapes
- `task.py` - Specifies the schema of the inputs and outputs for the problem




