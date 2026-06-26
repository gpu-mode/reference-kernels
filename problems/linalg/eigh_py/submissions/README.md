# Eigh Submission Attempts

These are durable smoke-test submissions for the `eigh` problem. They are not
part of the public starter template; they exist so evaluator changes can be
checked against a simple baseline and the fastest structured attempt found so
far.

- `torch_eigh.py`: direct dense eigensolver baseline.
- `triton_diagonal_fast_path.py`: exact diagonal fast path that uses Triton to
  materialize the permuted eigenvector basis, with dense fallback.

## Local KernelBot Measurements

Measured through a local KernelBot debug API against the final 39-test,
14-benchmark `eigh_py-dev` leaderboard on B200.

| Submission | Mode | Local submission | Result | Evaluator duration |
| --- | --- | ---: | --- | ---: |
| `torch_eigh.py` | test | 20 | 39/39 passed | 5.808s |
| `torch_eigh.py` | benchmark | 21 | 14/14 passed | 28.002s |
| `triton_diagonal_fast_path.py` | test | 22 | 39/39 passed | 5.626s |
| `triton_diagonal_fast_path.py` | benchmark | 23 | 14/14 passed | 45.949s |

On the final benchmark set, `triton_diagonal_fast_path.py` measured `1.618x`
geometric mean speedup over `torch_eigh.py`. The speedup comes from the
`batch=2,n=4096,case=diagonal` benchmark, where the Triton path was `1102.449x`
faster. Dense, mixed, rank-deficient, near-rank-deficient, clustered, and
LAPACK dense spectrum cases use the dense fallback path and are roughly neutral,
which is intentional benchmark signal.
