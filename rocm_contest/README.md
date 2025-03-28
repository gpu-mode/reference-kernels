# ROCM_CONTEST_KERNELS_REFERENCE

Welcome to the AMD ROCm Contest Kernels Reference repository! This repository focused on top3 DeepSeek-R1 kernels of Fused MOE, Fused MLA, Blockscale Gemm. Here you'll find clear instructions to test the triton kernels as your reference to help you figure out the design. Enjoy the contest.

## Repository Structure

The following code lists the directory structure:

```
.
└── op_tests
    ├── test_gemm_a8w8_blockscale.py
    ├── WIP FMOE
    ├── WIP FMLA
└── src
    ├── kernels
    │   ├── triton
    │     ├── a8w8_gemm
    │       ├── configs
    │       ├── __init__.py
    │       ├── blockwise_a8w8_gemm.py
    │     ├── WIP FMOE
    │     └── WIP FMLA
└── README.md
└── requirements.txt
└── utils.py

```

## Quick Start
Build a docker environment:
```bash
alias drun='docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v 
drun --name rocm_contest rocm/pytorch:latest
```
Clone the repository:
```bash
git clone git@github.com:ROCm/ROCM_CONTEST_KERNELS_REFERENCE.git
cd ROCM_CONTEST_KERNELS_REFERENCE.git
```
Prequisite installation:
```bash
pip install -r requirements.txt
```

## Run the Unit Test
Block Scale Gemm:
```bash
cd op_test
python test_gemm_a8w8_blockscale.py ## we compare triton block gemm with torch implementation
```
You will get:
```bash
...
a,b: [perf] dim: (16, 1536, 7168)     dtype: torch.bfloat16, torch avg: 193.94   us, triton avg: 75.05    us, uplift: 158.4%[checkAllclose atol=0.01 rtol=0.01 passed~]
a,b: [perf] dim: (16, 3072, 1536)     dtype: torch.bfloat16, torch avg: 103.46   us, triton avg: 15.80    us, uplift: 554.9%[checkAllclose atol=0.01 rtol=0.01 passed~]
a,b: [perf] dim: (16, 576, 7168)      dtype: torch.bfloat16, torch avg: 137.40   us, triton avg: 56.00    us, uplift: 145.3%[checkAllclose atol=0.01 rtol=0.01 passed~]
a,b: [perf] dim: (16, 7168, 256)      dtype: torch.bfloat16, torch avg: 54.82    us, triton avg: 6.15     us, uplift: 790.7%[checkAllclose atol=0.01 rtol=0.01 passed~]
a,b: [perf] dim: (16, 7168, 2048)     dtype: torch.bfloat16, torch avg: 205.79   us, triton avg: 25.95    us, uplift: 693.0%[checkAllclose atol=0.01 rtol=0.01 passed~]
...
```






