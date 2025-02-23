---
sidebar_position: 6
---

# Active Leaderboards
Here, will give leaderboard creators on the official leaderboard the option to provide more metadata and information
about their leaderboards. This may help reduce the friction between understanding the expected input
type of a leaderboard, and writing kernels. *This page will likely be constantly updating.*

## Introductory Leaderboards
These leaderboards are designed to help you get started, and were used as examples for previous
sections. They are not meant to be iterated on.
### Identity Kernel (Python)
* **leaderboard name**: `identity_py`
* **description**: Write a kernel that takes a list of tensors in PyTorch `List[torch.Tensor]` and returns an identical `List[torch.Tensor]`.

### Identity Kernel (CUDA)
* **leaderboard name**: `identity_cuda`
* **description**: Write a kernel that takes a list of tensors in memory `std::array<std::vector<float>, INT>` and returns an identical `std::array<std::vector<float>, INT>`.

## Practice Round Leaderboard
Most of these problems are derived from examples in the PMPP textbook.

### Conv2D Kernel
* **leaderboard name**: `conv2d`
* **description**: Given a pair of tensors `input: Tuple[torch.Tensor, torch.Tensor]`, Given an input tensor and a kernel tensor,
compute the 2D convolution of the input tensor about the kernel.
The data will be given to you **on-device** and expected to remain **on-device**, so there is no need to move from CPU to GPU.

### Grayscale Kernel
* **leaderboard name**: `grayscale`
* **description**: Given an RGB `torch.Tensor` of shape (H, W, 3) with values in [0, 1], compute the grayscale conversion.
The data will be given to you **on-device** and expected to remain **on-device**, so there is no need to move from CPU to GPU.


### Histogram Kernel
* **leaderboard name**: `histogram`
* **description**: Given a `torch.Tensor` of shape * **description**: Given an RGB `torch.Tensor` of shape `(size,)`, compute a histogram.
The data will be given to you **on-device** and expected to remain **on-device**, so there is no need to move from CPU to GPU.


### Matmul Kernel
* **leaderboard name**: `matmul`
* **description**: Given a pair of matrices `input: Tuple[torch.Tensor, torch.Tensor]`, compute their multiplication.
The data will be given to you **on-device** and expected to remain **on-device**, so there is no need to move from CPU to GPU.

### Prefix-Sum Kernel
* **leaderboard name**: `prefixsum`
* **description**: Given a 1D tensor `torch.Tensor`, compute the prefix sum.
The data will be given to you **on-device** and expected to remain **on-device**, so there is no need to move from CPU to GPU.

### Sort Kernel
* **leaderboard name**: `prefixsum`
* **description**: Given a 1D tensor `torch.Tensor`, sort the tensor and return the sorted tensor.
The data will be given to you **on-device** and expected to remain **on-device**, so there is no need to move from CPU to GPU.

### Vector-Add Kernel
* **leaderboard name**: `vectoradd`
* **description**: Given a pair of tensors `input: Tuple[torch.Tensor, torch.Tensor]`, add the two tensors and return it.
The data will be given to you **on-device** and expected to remain **on-device**, so there is no need to move from CPU to GPU.

### Vector-Sum Kernel
* **leaderboard name**: `vectorsum`
* **description**: Given a pair of tensors `torch.Tensor`, compute the element-wise sum.
The data will be given to you **on-device** and expected to remain **on-device**, so there is no need to move from CPU to GPU.
