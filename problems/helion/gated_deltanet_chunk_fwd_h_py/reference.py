import torch
import torch.nn.functional as F
from task import input_t, output_t
from utils import verbose_allclose

CHUNK_SIZE = 64

# Use FLA's Triton kernels as reference (same Triton tl.dot as Helion)
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h as fla_chunk_fwd_h
from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd as fla_recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum, solve_tril
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd


def generate_input(B: int, T: int, H: int, K: int, V: int, seed: int) -> input_t:
    torch.manual_seed(seed)
    device = "cuda"
    # Generate pipeline-derived inputs: base inputs -> g_cumsum, A, w, u via FLA utilities
    k = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32, device=device), p=2, dim=-1)
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    g = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    g_cumsum = chunk_local_cumsum(g, chunk_size=CHUNK_SIZE)
    A = chunk_scaled_dot_kkt_fwd(k=k, g=g_cumsum, beta=beta, output_dtype=torch.float32)
    A = solve_tril(A=A, output_dtype=k.dtype)
    w, u = fla_recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g=g_cumsum)
    return k.contiguous(), w.contiguous(), u.contiguous(), g_cumsum.contiguous()


def ref_kernel(data: input_t) -> output_t:
    k, w, u, g = data
    h, v_new, _ = fla_chunk_fwd_h(
        k=k, w=w, u=u, g=g,
        initial_state=None,
        output_final_state=False,
    )
    return h, v_new


def check_implementation(data, output):
    expected = ref_kernel(data)
    exp_h, exp_v = expected
    got_h, got_v = output

    reasons_h = verbose_allclose(got_h.float(), exp_h.float(), rtol=1e-2, atol=1e-2)
    reasons_v = verbose_allclose(got_v.float(), exp_v.float(), rtol=1e-2, atol=1e-2)

    reasons = []
    if reasons_h:
        reasons.append("h mismatch: " + " ".join(reasons_h))
    if reasons_v:
        reasons.append("v_new mismatch: " + " ".join(reasons_v))

    if reasons:
        return False, " | ".join(reasons)
    return True, ""
