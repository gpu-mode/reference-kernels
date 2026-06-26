# Eigh Submission Attempts

These are durable smoke-test submissions for the `eigh` problem. They are not
part of the public starter template; they exist so evaluator changes can be
checked against a simple baseline and the fastest structured attempt found so
far.

- `torch_eigh.py`: direct dense eigensolver baseline.
- `triton_diagonal_fast_path.py`: exact diagonal fast path that uses Triton to
  materialize the permuted eigenvector basis, with dense fallback.

## Local KernelBot Measurements

Measured through the local KernelBot API with `popcorn-cli submit --gpu B200`.
Speedup is relative to `torch_eigh.py`.

| Submission | Mean ns | Speedup |
| --- | ---: | ---: |
| `torch_eigh.py` | 61,283,134.5 | 1.000x |
| `triton_diagonal_fast_path.py` | 187,568.5 | 326.724x |

This table is from a temporary `eigh_diag_measure` leaderboard using the
official large diagonal shape `batch=1,n=4096,case=diagonal`.

On the full local KernelBot `eigh` benchmark list, `triton_diagonal_fast_path.py`
passed as submission 11 and measured `1.590x` geometric mean speedup over
`torch_eigh.py` submission 10. It was approximately neutral on dense fallback
cases and `330.243x` faster on the included `batch=1,n=4096,case=diagonal`
benchmark.

`triton_diagonal_fast_path.py` also passed local KernelBot test mode on a temp
test set containing both diagonal and dense fallback cases.
