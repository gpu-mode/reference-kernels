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

The expanded benchmark list includes additional spectrum, PSD, repeated,
banded, and row-scaled cases where `triton_diagonal_fast_path.py` falls back to
the dense PyTorch path. Those fallback costs are intentional benchmark signal:
the diagonal specialization should win only on the diagonal case and stay close
to baseline elsewhere.

On the expanded local KernelBot `eigh` benchmark list,
`triton_diagonal_fast_path.py` passed as submission 13 and measured `1.305x`
geometric mean speedup over `torch_eigh.py` submission 12. It was `320.003x`
faster on the included `batch=1,n=4096,case=diagonal` benchmark and roughly
neutral on the additional dense fallback cases.

Both retained submissions also passed local KernelBot test mode over all 39
test specs: `torch_eigh.py` as submission 14 and
`triton_diagonal_fast_path.py` as submission 15.
