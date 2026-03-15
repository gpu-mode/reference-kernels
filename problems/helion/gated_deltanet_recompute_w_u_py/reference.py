"""
Gated DeltaNet: chunk-wise recomputation of w and u
====================================================

Background
----------
DeltaNet is a linear recurrent model based on the **delta rule** (Widrow-Hoff,
1960), which updates a matrix-valued hidden state H_t ∈ R^{K×V} as:

    H_t = exp(g_t) · H_{t-1}  +  β_t · (v_t − H_{t-1} k_t) ⊗ k_t

where:
  k_t ∈ R^K        — key (query used for the associative read)
  v_t ∈ R^V        — value (target to associate)
  β_t ∈ (0,1)      — per-step learning rate / write strength
  g_t ≤ 0          — log-decay (gate); exp(g_t) ∈ (0,1] damps old memories
  ⊗                — outer product

The exp(g_t) term makes this the **Gated DeltaNet** (Yang et al., 2024).
The delta rule term `−H_{t-1} k_t ⊗ k_t` is an error-correction: it first
reads what the current memory says about k_t, then erases it before writing
v_t, so the net update is proportional to the prediction error (v_t − ŷ_t).

Chunk-wise parallel form
------------------------
To exploit GPU parallelism, the sequence of length T is split into NT = T/C
non-overlapping chunks of size C. Within chunk n (positions t = nC … nC+C−1)
define τ = t − nC as the within-chunk index.

The chunk boundary state update can be written as:

    H_{n+1} = exp(G_n) · H_n  +  Σ_τ  u_τ ⊗ k_τ  −  w_τ ⊗ k_τ · H_n
            = exp(G_n) · H_n  +  (U_n  −  W_n) · H_n          (matrix form)

where G_n = Σ_τ g_{nC+τ} is the total chunk decay.  The matrices U_n and W_n
collect the within-chunk "value writes" and "key absorptions" respectively,
and their efficient computation is the purpose of this file.

Intra-chunk interaction matrix M
---------------------------------
For positions i > j within the same chunk (strict causal order), position j's
delta-rule write creates a residual that position i must account for:

    M_{i,j} = β_i · exp(g̃_i − g̃_j) · (k_i k_j^T)    (i > j, else 0)

where g̃_t is the **chunk-local** cumulative sum of g (restarted at each chunk
boundary).  The exponential exp(g̃_i − g̃_j) is the decay from position j to
position i within the chunk.

The k_i k_j^T factor arises because the delta rule at step i reads k_i^T H,
and H already contains the write β_j k_j ⊗ k_j from step j, contributing
β_j (k_i^T k_j) k_j to the correction term.

Solving for A = (I + M)^{-1}
------------------------------
Within a chunk the delta-rule corrections chain together: the write at step j
is partially cancelled by step j+1, which is further modified by step j+2,
etc.  Summing this geometric-series of interactions leads to a linear system:

    (I + M) · A = I    ⟹    A = (I + M)^{-1}

Because M is **strictly lower-triangular**, (I + M) is unit lower-triangular
and the solve is numerically stable.  A is computed once per chunk and reused
for both w and u.

Computing w and u
-----------------
Given the resolved interaction matrix A = (I + M)^{-1}:

    u_i = Σ_j  A_{i,j} · β_j · v_j          (shape: [K])
    w_i = Σ_j  A_{i,j} · β_j · exp(g̃_j) · k_j   (shape: [V])

In matrix form (over the C positions in a chunk):

    U = A · (β ⊙ v)                  (u_c in the code)
    W = A · (β ⊙ exp(g̃) ⊙ k)        (w_c in the code)

  u accumulates the net value contributions reaching each position after all
  within-chunk delta-rule cancellations have been applied.

  w accumulates the net key absorptions; the exp(g̃_j) factor converts the
  local-cumsum decay at j into the correct scale for the outer chunk loop
  (where H_{n,0} is decayed by the full chunk decay to position i).

These (w, u) pairs are the inputs consumed by the chunk-wise outer state-
transition loop of the full Gated DeltaNet forward pass.

References
----------
- Yang et al. "Gated Delta Networks: Improving Mamba2 with Delta Rule" (2024)
  https://arxiv.org/abs/2412.06464
- Widrow & Hoff. "Adaptive Switching Circuits" (1960) — original delta rule
- Schmidhuber. "Learning to Control Fast-Weight Memories" (1992) — fast-weight
  programmers; modern framing of outer-product / delta-rule recurrences
- Sun et al. "Retentive Network" / "Flash Linear Attention" — chunk-wise
  formulation of linear RNN state transitions used here
"""
import torch
from task import input_t, output_t
from utils import verbose_allclose

CHUNK_SIZE = 64


def _chunk_local_cumsum_eager(g, chunk_size):
    B, T, H = g.shape
    C = chunk_size
    return g.float().reshape(B, T // C, C, H).cumsum(dim=2).reshape(B, T, H)


def _chunk_scaled_dot_kkt_fwd_eager(k, g_cumsum, beta, chunk_size):
    B, T, H, K = k.shape
    C = chunk_size
    NT = T // C
    k_c = k.float().reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4)
    g_c = g_cumsum.float().reshape(B, NT, C, H).permute(0, 1, 3, 2)
    beta_c = beta.float().reshape(B, NT, C, H).permute(0, 1, 3, 2)
    kkt = k_c @ k_c.transpose(-1, -2)
    strict_lower = torch.tril(torch.ones(C, C, device=k.device), diagonal=-1)
    g_diff = g_c.unsqueeze(-1) - g_c.unsqueeze(-2)
    g_diff = g_diff * strict_lower
    A = kkt * beta_c.unsqueeze(-1) * torch.exp(g_diff) * strict_lower
    return A.permute(0, 1, 3, 2, 4).reshape(B, T, H, C).to(torch.float32)


def _solve_tril_eager(A, output_dtype):
    B, T, H, C = A.shape
    NT = T // C
    A_mat = A.float().reshape(B, NT, C, H, C).permute(0, 1, 3, 2, 4)
    eye = torch.eye(C, device=A.device).expand_as(A_mat)
    result = torch.linalg.solve_triangular(eye + A_mat, eye, upper=False)
    return result.permute(0, 1, 3, 2, 4).reshape(B, T, H, C).to(output_dtype)


def generate_input(B: int, T: int, H: int, K: int, V: int, seed: int) -> input_t:
    torch.manual_seed(seed)
    device = "cuda"
    k = torch.randn(B, T, H, K, dtype=torch.float32, device=device) / K**0.5
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    g_inc = -torch.abs(torch.randn(B, T, H, dtype=torch.float32, device=device))
    g = g_inc.cumsum(dim=1)
    g_cumsum = _chunk_local_cumsum_eager(g, chunk_size=CHUNK_SIZE)
    A = _chunk_scaled_dot_kkt_fwd_eager(k=k, g_cumsum=g_cumsum, beta=beta, chunk_size=CHUNK_SIZE)
    A = _solve_tril_eager(A=A, output_dtype=k.dtype)
    return k.contiguous(), v.contiguous(), beta.contiguous(), A.contiguous(), g_cumsum.contiguous()


def ref_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    C = A.shape[-1]
    NT = T // C
    k_c = k.float().reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4)
    v_c = v.float().reshape(B, NT, C, H, V).permute(0, 1, 3, 2, 4)
    beta_c = beta.float().reshape(B, NT, C, H).permute(0, 1, 3, 2)
    g_c = g.float().reshape(B, NT, C, H).permute(0, 1, 3, 2)
    A_c = A.float().reshape(B, NT, C, H, C).permute(0, 1, 3, 2, 4)
    u_c = A_c @ (v_c * beta_c.unsqueeze(-1))
    w_c = A_c @ (k_c * (beta_c * torch.exp(g_c)).unsqueeze(-1))
    w = w_c.permute(0, 1, 3, 2, 4).reshape(B, T, H, K).to(k.dtype)
    u = u_c.permute(0, 1, 3, 2, 4).reshape(B, T, H, V).to(v.dtype)
    return w, u


def check_implementation(data, output):
    expected = ref_kernel(data)
    exp_w, exp_u = expected
    got_w, got_u = output

    reasons_w = verbose_allclose(got_w, exp_w, rtol=1e-3, atol=1e-3)
    reasons_u = verbose_allclose(got_u, exp_u, rtol=1e-3, atol=1e-3)

    reasons = []
    if reasons_w:
        reasons.append("w mismatch: " + " ".join(reasons_w))
    if reasons_u:
        reasons.append("u mismatch: " + " ".join(reasons_u))

    if reasons:
        return False, " | ".join(reasons)
    return True, ""
