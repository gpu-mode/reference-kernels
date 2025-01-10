---
sidebar_position: 1
---

# Getting Started
Welcome! If you are excited about building GPU kernels, this leaderboard is the place for you! We
have designed a fully-functional competition platform on [Discord](https://discord.com/), where users can **submit their own
kernel implementations** for popular algorithms (e.g. matrix multiplication, softmax, [Llama 3](https://ai.meta.com/blog/meta-llama-3/) inference, etc.) and compete to produce the fastest kernels.

Users interface directly with a Discord bot, which runs and measures various metrics for their
kernel submissions on our own cloud compute, so **participants can compete without their own GPUs**!
We are gracious to our sponsors for providing these community resources.

## Why a GPU Kernel Leaderboard?

Large foundation models (e.g. [language models](https://ai.meta.com/blog/meta-llama-3/), [protein models](https://www.evolutionaryscale.ai/blog/esm3-release), etc.) use an extraordinary 
amount of compute resources to both train and run inference. These models are ultimately built on
top of deep learning libraries like [PyTorch](https://pytorch.org/) and
[Jax](https://jax.readthedocs.io/en/latest/quickstart.html) that dispatch kernels to run quickly on
GPUs. Unfortunately, many of these kernels are far from optimal due to the steep learning curve and
lack of abundant resources for learning how to write these kernels. Even tools like
[`torch.compile`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) that try to automate kernel optimization are far from optimal.

Recent efforts such as [Liger Kernel](https://github.com/linkedin/Liger-Kernel) and the [GPU MODE
community](https://www.youtube.com/channel/UCJgIbYl6C5no72a0NUAPcTA) have been built around making
these kernels more accessible, and this leaderboard is another step in that direction. Unlike
standard algorithm development, GPU kernel development involves optimizing **a particular
algorithm** on **a particular set of inputs** on a **particular device**. Even for a fixed
algorithm, there is a combinatorially large number of implementations that are "useful". We designed
this leaderboard as a central and open-source resource for people to find the fastest kernels for
the devices they are using. Furthermore, these open-community kernels will be useful in the future for
designing automated methods for optimized kernel generation.

## How the Leaderboard Works
The leaderboard is designed under the principle that an optimized kernel for algorithm `X` on device
`A` might not be optimized on device `B`. Thus, we separate runtime rankings entirely for the same
algorithm on different devices. Furthermore, certain algorithms like `Llama 3` inference probably
will not fit in-memory on, say an `NVIDIA T4`, so we allow leaderboard creators to specify which devices are
available for their leaderboard.

We built the entire leaderboard interface around a **Discord bot** that connects to a set of runners ([GitHub Actions](https://github.com/features/actions) and
[Modal](https://modal.com/) currently) with their own cloud compute. In other words, participants can write kernels
**without access to their own GPUs**! The leaderboard will be hosted on the GPU MODE discord, where
participant can submit and run their kernels for free. The bot also provides mechanisms for
debugging kernels, and we plan to implement profiling tools such as
[`ncu`](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) in the future as well.

## Language Support
We aim to support as many languages, DSLs, and frameworks for GPU programming as possible. We also
do not want to limit participants to NVIDIA GPUs. Currently, we have tested the leaderboard to allow
submissions using:

#### On Python Leaderboards
* PyTorch
* Jax
* Triton
* Inline CUDA

#### On CUDA Leaderboards
* CUDA
* CuBLAS
* CUTLASS
* ThunderKittens

## Participating
We have tried to make the platform as intuitive as possible, but we are open to feedback. You are
now ready to submit your first leaderboard kernel -- go to the next section to begin!


