import torch
import math
from task import input_t, output_t
from utils import make_match_reference, DeterministicContext


def generate_input(B: int, H: int, S: int, d_c: int, d_r: int, seed: int) -> input_t:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    q_nope = torch.randn(B, H, d_c, dtype=torch.float16, device="cuda", generator=gen).contiguous()
    q_pe = torch.randn(B, H, d_r, dtype=torch.float16, device="cuda", generator=gen).contiguous()
    kv_c = torch.randn(B, S, d_c, dtype=torch.float16, device="cuda", generator=gen).contiguous()
    k_pe = torch.randn(B, S, d_r, dtype=torch.float16, device="cuda", generator=gen).contiguous()
    return q_nope, q_pe, kv_c, k_pe


def ref_kernel(data: input_t) -> output_t:
    with DeterministicContext():
        q_nope, q_pe, kv_c, k_pe = data
        B, H, d_c = q_nope.shape
        d_r = q_pe.shape[-1]
        sm_scale = 1.0 / math.sqrt(d_c + d_r)

        # Content score: [B, H, S] = [B, H, d_c] @ [B, d_c, S]
        score_content = torch.bmm(q_nope.float(), kv_c.float().transpose(-2, -1))

        # Position score: [B, H, S] = [B, H, d_r] @ [B, d_r, S]
        score_position = torch.bmm(q_pe.float(), k_pe.float().transpose(-2, -1))

        # Combined score with scaling
        scores = (score_content + score_position) * sm_scale

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Output: [B, H, d_c] = [B, H, S] @ [B, S, d_c]
        output = torch.bmm(attn_weights, kv_c.float())

        return output.to(q_nope.dtype)


check_implementation = make_match_reference(ref_kernel, rtol=1e-2, atol=1e-2)
