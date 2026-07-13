# Cholesky baselines

- `torch_cholesky_ex.py` uses the non-throwing `torch.linalg.cholesky_ex` path
  and is the starter baseline.
- `torch_cholesky.py` keeps the synchronizing `torch.linalg.cholesky` path for
  comparison.
- `blocked_torch_256.py` and `blocked_torch_512.py` are simple blocked
  Cholesky experiments built from panel factorizations, triangular solves, and
  trailing batched matrix updates.
- `blocked_torch_1024.py` explores a wider panel, while
  `blocked_torch_auto.py` uses the best simple strategy found by the initial
  B200 sweep.
- `triton_cholesky32.py` is a guard-compatible Triton experiment for the
  heavily batched 32 x 32 case, with `torch.linalg.cholesky_ex` fallback for
  the remaining dimensions.

Every file implements the public `custom_kernel(data)` contract and is useful
for checking evaluator overhead before testing custom kernels.
