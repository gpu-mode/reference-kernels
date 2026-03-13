import torch
import torch.nn.functional as F
from task import input_t, output_t
from utils import make_match_reference

CHUNK_SIZE = 64

# Use FLA's Triton kernels as reference (same Triton tl.dot as Helion)
from fla.ops.common.chunk_o import chunk_fwd_o as fla_chunk_fwd_o
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h as fla_chunk_fwd_h
from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd as fla_recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum, solve_tril
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd


def generate_input(B: int, T: int, H: int, K: int, V: int, seed: int) -> input_t:
    torch.manual_seed(seed)
    device = "cuda"
    # Generate pipeline-derived inputs: base inputs -> g_cumsum, A, w, u, h, v_new via FLA utilities
    q = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    k = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32, device=device), p=2, dim=-1)
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    g = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    g_cumsum = chunk_local_cumsum(g, chunk_size=CHUNK_SIZE)
    A = chunk_scaled_dot_kkt_fwd(k=k, g=g_cumsum, beta=beta, output_dtype=torch.float32)
    A = solve_tril(A=A, output_dtype=k.dtype)
    w, u = fla_recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g=g_cumsum)
    h, v_new, _ = fla_chunk_fwd_h(k=k, w=w, u=u, g=g_cumsum, output_final_state=False)
    return q.contiguous(), k.contiguous(), v_new.contiguous(), h.contiguous(), g_cumsum.contiguous()


def ref_kernel(data: input_t) -> output_t:
    q, k, v_new, h, g = data
    K = q.shape[-1]
    scale = K ** -0.5
    o = fla_chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale)
    return o


check_implementation = make_match_reference(ref_kernel, rtol=1e-2, atol=1e-2)
