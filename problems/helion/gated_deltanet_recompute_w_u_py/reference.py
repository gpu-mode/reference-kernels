import torch
import torch.nn.functional as F
from task import input_t, output_t
from utils import verbose_allclose

CHUNK_SIZE = 64

# Use FLA's Triton kernels as reference (same Triton tl.dot as Helion)
from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd as fla_recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum, solve_tril
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd


def generate_input(B: int, T: int, H: int, K: int, V: int, seed: int) -> input_t:
    torch.manual_seed(seed)
    device = "cuda"
    # Generate pipeline-derived inputs: base inputs -> g_cumsum, A via FLA utilities
    k = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32, device=device), p=2, dim=-1)
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    g = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    g_cumsum = chunk_local_cumsum(g, chunk_size=CHUNK_SIZE)
    A = chunk_scaled_dot_kkt_fwd(k=k, g=g_cumsum, beta=beta, output_dtype=torch.float32)
    A = solve_tril(A=A, output_dtype=k.dtype)
    return k.contiguous(), v.contiguous(), beta.contiguous(), A.contiguous(), g_cumsum.contiguous()


def ref_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    w, u = fla_recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g=g)
    return w, u


def check_implementation(data, output):
    expected = ref_kernel(data)
    exp_w, exp_u = expected
    got_w, got_u = output

    reasons_w = verbose_allclose(got_w, exp_w, rtol=1e-2, atol=1e-2)
    reasons_u = verbose_allclose(got_u, exp_u, rtol=1e-2, atol=1e-2)

    reasons = []
    if reasons_w:
        reasons.append("w mismatch: " + " ".join(reasons_w))
    if reasons_u:
        reasons.append("u mismatch: " + " ".join(reasons_u))

    if reasons:
        return False, " | ".join(reasons)
    return True, ""
