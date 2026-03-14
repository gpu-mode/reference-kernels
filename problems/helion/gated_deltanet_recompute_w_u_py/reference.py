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
