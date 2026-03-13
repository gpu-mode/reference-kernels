import torch
import torch.nn.functional as F
from task import input_t, output_t
from utils import make_match_reference

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
    g_diff = g_c.unsqueeze(-1) - g_c.unsqueeze(-2)
    strict_lower = torch.tril(torch.ones(C, C, device=k.device), diagonal=-1)
    A = kkt * beta_c.unsqueeze(-1) * torch.exp(g_diff) * strict_lower
    return A.permute(0, 1, 3, 2, 4).reshape(B, T, H, C).to(torch.float32)


def _solve_tril_eager(A, output_dtype):
    B, T, H, C = A.shape
    NT = T // C
    A_mat = A.float().reshape(B, NT, C, H, C).permute(0, 1, 3, 2, 4)
    eye = torch.eye(C, device=A.device).expand_as(A_mat)
    result = torch.linalg.solve_triangular(eye + A_mat, eye, upper=False)
    return result.permute(0, 1, 3, 2, 4).reshape(B, T, H, C).to(output_dtype)


def _recompute_w_u_fwd_eager(k, v, beta, A, g):
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


def _chunk_fwd_h_eager(k, w, u, g):
    B, T, H, K = k.shape
    V = u.shape[-1]
    C = CHUNK_SIZE
    NT = T // C
    k_c = k.float().reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4)
    w_c = w.float().reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4)
    u_c = u.float().reshape(B, NT, C, H, V).permute(0, 1, 3, 2, 4)
    g_c = g.float().reshape(B, NT, C, H).permute(0, 1, 3, 2)
    h_all = torch.zeros(B, NT, H, K, V, dtype=torch.float32, device=k.device)
    v_new_c = torch.zeros_like(u_c)
    h = torch.zeros(B, H, K, V, dtype=torch.float32, device=k.device)
    for c in range(NT):
        h_all[:, c] = h
        v_new_c[:, c] = u_c[:, c] - w_c[:, c] @ h
        g_last = g_c[:, c, :, -1]
        gate = torch.exp(g_last.unsqueeze(-1) - g_c[:, c])
        v_gated = v_new_c[:, c] * gate.unsqueeze(-1)
        h = h * torch.exp(g_last).unsqueeze(-1).unsqueeze(-1) + k_c[:, c].transpose(-1, -2) @ v_gated
    v_new_out = v_new_c.permute(0, 1, 3, 2, 4).reshape(B, T, H, V).to(u.dtype)
    return h_all.to(k.dtype), v_new_out


def generate_input(B: int, T: int, H: int, K: int, V: int, seed: int) -> input_t:
    torch.manual_seed(seed)
    device = "cuda"
    q = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    k = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32, device=device), p=2, dim=-1)
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    g = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    g_cumsum = _chunk_local_cumsum_eager(g, chunk_size=CHUNK_SIZE)
    A = _chunk_scaled_dot_kkt_fwd_eager(k=k, g_cumsum=g_cumsum, beta=beta, chunk_size=CHUNK_SIZE)
    A = _solve_tril_eager(A=A, output_dtype=k.dtype)
    w, u = _recompute_w_u_fwd_eager(k=k, v=v, beta=beta, A=A, g=g_cumsum)
    h, v_new = _chunk_fwd_h_eager(k=k, w=w, u=u, g=g_cumsum)
    return q.contiguous(), k.contiguous(), v_new.contiguous(), h.contiguous(), g_cumsum.contiguous()


def ref_kernel(data: input_t) -> output_t:
    q, k, v_new, h, g = data
    B, T, H, K = q.shape
    V = v_new.shape[-1]
    C = CHUNK_SIZE
    NT = T // C
    scale = K ** -0.5
    q_c = q.float().reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4)
    k_c = k.float().reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4)
    v_c = v_new.float().reshape(B, NT, C, H, V).permute(0, 1, 3, 2, 4)
    g_c = g.float().reshape(B, NT, C, H).permute(0, 1, 3, 2)
    o_inter = (q_c @ h.float()) * torch.exp(g_c).unsqueeze(-1)
    qk = q_c @ k_c.transpose(-1, -2) * torch.exp(g_c.unsqueeze(-1) - g_c.unsqueeze(-2))
    causal = torch.tril(torch.ones(C, C, device=q.device))
    o = (o_inter + (qk * causal) @ v_c) * scale
    return o.permute(0, 1, 3, 2, 4).reshape(B, T, H, V).to(q.dtype)


check_implementation = make_match_reference(ref_kernel, rtol=1e-2, atol=1e-2)
