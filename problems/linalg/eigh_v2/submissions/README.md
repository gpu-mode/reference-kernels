# Eigh v2 Submission Attempts

These are durable smoke-test submissions for the `eigh_v2` problem. They are not
part of the public starter template; they exist so evaluator changes can be
checked against a simple baseline and the fastest structured attempt found so
far.

- `torch_eigh.py`: direct dense eigensolver baseline.
- `triton_diagonal_fast_path.py`: exact diagonal fast path that uses Triton to
  materialize the permuted eigenvector basis, with dense fallback.

## Local KernelBot Measurements

Regenerate measurements against the `eigh_v2` leaderboard before quoting
speedups. The retained Triton submission remains a structured smoke test: it
validates that diagonal fast paths can pass correctness, while dense, mixed,
rank-deficient, clustered, row-scaled, block-diagonal, and LAPACK dense spectrum
cases use the dense fallback path.
