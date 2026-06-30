# QR v2 conditioning hardening

QR v2 intentionally allows implementations to choose different internal
precision strategies. The correctness contract is about the returned FP32
compact Householder factors, not about forbidding FP16, FP8, NVFP4, or
shape-specific dispatch inside the kernel.

The hardening target is narrower: do not let the benchmark reward a solution
that picks precision only from the public shape and ignores the numerical
quality of the actual matrices in that shape.

## Pattern

Use the same public shape with several matrix distributions:

- `dense` for the common well-conditioned path.
- `rankdef`, `nearrank`, `clustered`, `band`, `rowscale`, and
  `nearcollinear` for numerical stress.
- `mixed` for heterogeneous batches where each matrix independently receives a
  profile at a random batch position.

This means a `batch x 512 x 512` benchmark can contain easy and hard matrices in
the same call. A submission can still specialize for `n = 512`, but it has to
either use a robust path for the whole batch or inspect matrix quality before
routing individual work. Pure shape-only low-precision routing is much less
likely to pass, and pure shape-only high-precision routing pays the ranked cost
on easy matrices.

## Smoke checks

The `tests:` list is useful for local iteration, but it should be treated as a
smoke path. Contestants can run it before submitting, and some workflows may use
it as a fast correctness check, but it is not enough to enforce the competition
contract by itself.

Do not rely on `tests:` to prevent shape-only precision routing. Any behavior
that must affect eligibility or rank needs to appear in `benchmarks:`, because
leaderboard mode validates benchmark outputs.

## Ranked checks

The `benchmarks:` list includes both:

- production-like dense shapes, so fast low-precision paths still matter when
  they are numerically valid; and
- mixed or homogeneous stress shapes, so robustness is part of the score rather
  than only a pass/fail gate.

QR v2 includes ranked `512 x 512` stress benchmarks for `mixed`, `rankdef`,
`clustered`, `rowscale`, and `nearcollinear` distributions. Those cases are the
main guard against routing precision purely from the public shape.

The evaluator combines each public seed with `POPCORN_SEED` when the service
sets it. That keeps runs reproducible while preventing exact input matrices from
being fully determined by the public `task.yml`.

## Extending the suite

When adding a new QR shape or distribution, prefer adding it in pairs:

1. A representative `benchmarks:` case when the behavior should affect
   eligibility or ranking.
2. An optional cheap `tests:` smoke case if it helps local iteration.

For compute-heavy stress distributions, prefer one mixed benchmark before adding
many homogeneous stress cases. If a homogeneous case is important enough to
change the competition outcome, put it in `benchmarks:` first and only then add
a smaller `tests:` proxy for convenience.
