---
sidebar_position: 5
---

# Active Leaderboards
Here, will give leaderboard creators on the official leaderboard the option to provide more metadata and information
about their leaderboards. This may help reduce the friction between understanding the expected input
type of a leaderboard, and writing kernels. *This page will likely be constantly updating.*


## Python Leaderboards

### Softmax Kernel
* **leaderboard name**: `softmax-alpha-test`
* **description**: Given a list of input tensors `input: List[torch.Tensor]`, compute the softmax along the last
dimension `dim=-1` of each tensor in the list. Return this list of tensors as `List[torch.Tensor]`.
The data will be given to you **on-device** and expected to remain **on-device**, so there is no need to move from CPU to GPU.

## CUDA Leaderboards
WIP.

## Introductory Leaderboards
These leaderboards are designed to help you get started, and were used as examples for previous
sections. They are not meant to be iterated on.
### Identity Kernel (Python)
* **leaderboard name**: `identity_py`
* **description**: Write a kernel that takes a list of tensors in PyTorch `List[torch.Tensor]` and returns an identical `List[torch.Tensor]`.

### Identity Kernel (CUDA)
* **leaderboard name**: `identity_cuda`
* **description**: Write a kernel that takes a list of tensors in memory `std::array<std::vector<float>, INT>` and returns an identical `std::array<std::vector<float>, INT>`.
