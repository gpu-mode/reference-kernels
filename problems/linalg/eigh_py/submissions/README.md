# Eigh Submission Attempts

These are durable smoke-test submissions for the `eigh` problem. They are not
part of the public starter template; they exist so evaluator changes can be
checked against a simple baseline and the fastest structured attempt found so
far.

- `torch_eigh.py`: direct dense eigensolver baseline.
- `triton_diagonal_fast_path.py`: exact diagonal fast path that uses Triton to
  materialize the permuted eigenvector basis, with dense fallback.

## Local KernelBot Measurements

Regenerate measurements whenever `task.yml` benchmark cases change. The current
benchmark set keeps dense, mixed, rank-deficient, near-rank-deficient,
clustered, diagonal, and two LAPACK dense spectrum cases. The Triton submission
is expected to win on the diagonal case and fall back to the dense PyTorch path
elsewhere; those fallback costs are intentional benchmark signal.
