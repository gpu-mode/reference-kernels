# Cholesky example submission

`triton_cholesky32.py` is a guard-compatible Triton implementation for the
heavily batched 32 x 32 case, with `torch.linalg.cholesky_ex` fallback for the
remaining dimensions. It implements the public `custom_kernel(data)` contract.
