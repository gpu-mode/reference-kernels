import torch
from task import input_t, output_t
from utils import make_match_reference, DeterministicContext


def generate_input(B: int, T: int, H: int, K: int, V: int, seed: int) -> input_t:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    q = torch.randn(B, T, H, K, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    k = torch.randn(B, T, H, K, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    # L2-normalize k along last dim
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    v = torch.randn(B, T, H, V, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    g = torch.randn(B, T, H, K, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32, device="cuda", generator=gen)).contiguous()
    return q, k, v, g, beta


def ref_kernel(data: input_t) -> output_t:
    with DeterministicContext():
        q, k, v, g, beta = data
        B, T, H, K = q.shape
        V_dim = v.shape[-1]
        scale = K ** -0.5

        # Reshape for batched matmul: [B*H, T, K/V]
        q_r = q.permute(0, 2, 1, 3).reshape(B * H, T, K)
        k_r = k.permute(0, 2, 1, 3).reshape(B * H, T, K)
        v_r = v.permute(0, 2, 1, 3).reshape(B * H, T, V_dim)
        g_r = g.permute(0, 2, 1, 3).reshape(B * H, T, K)
        beta_r = beta.permute(0, 2, 1).reshape(B * H, T)

        # Initialize hidden state [B*H, K, V]
        S = torch.zeros(B * H, K, V_dim, dtype=torch.float32, device=q.device)
        outputs = []

        for t in range(T):
            # Per-channel decay
            decay = torch.exp(g_r[:, t, :])  # [B*H, K]
            S = S * decay.unsqueeze(-1)  # [B*H, K, V]

            # Prediction: k_t^T @ S -> [B*H, V]
            k_t = k_r[:, t, :]  # [B*H, K]
            predicted = torch.bmm(k_t.unsqueeze(1), S).squeeze(1)  # [B*H, V]

            # Delta: v_t - predicted
            v_t = v_r[:, t, :]  # [B*H, V]
            delta = v_t - predicted  # [B*H, V]

            # Correction: S += k_t @ (beta_t * delta)^T
            b_t = beta_r[:, t].unsqueeze(-1)  # [B*H, 1]
            correction = torch.bmm(k_t.unsqueeze(-1), (b_t * delta).unsqueeze(1))  # [B*H, K, V]
            S = S + correction

            # Output: scale * q_t @ S
            q_t = q_r[:, t, :]  # [B*H, K]
            o_t = scale * torch.bmm(q_t.unsqueeze(1), S).squeeze(1)  # [B*H, V]
            outputs.append(o_t)

        # Stack and reshape: [B*H, T, V] -> [B, T, H, V]
        output = torch.stack(outputs, dim=1)
        output = output.reshape(B, H, T, V_dim).permute(0, 2, 1, 3).contiguous()
        return output


check_implementation = make_match_reference(ref_kernel, rtol=1e-3, atol=1e-3)
