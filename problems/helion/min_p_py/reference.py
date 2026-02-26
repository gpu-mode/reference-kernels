import torch
from task import input_t, output_t
from utils import make_match_reference, DeterministicContext


def generate_input(batch_size: int, vocab_size: int, seed: int) -> input_t:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    logits = torch.randn(batch_size, vocab_size, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    # min_p values between 0.05 and 0.2
    min_p = (torch.rand(batch_size, device="cuda", generator=gen) * 0.15 + 0.05).to(torch.float32).contiguous()
    return logits, min_p


def ref_kernel(data: input_t) -> output_t:
    with DeterministicContext():
        logits, min_p = data
        logits = logits.float()

        # 1. Softmax
        probs = torch.softmax(logits, dim=-1)

        # 2. Find max prob and compute threshold
        max_probs = probs.max(dim=-1, keepdim=True).values
        threshold = min_p[:, None] * max_probs

        # 3. Apply threshold
        filtered = torch.where(probs >= threshold, probs, torch.zeros_like(probs))

        # 4. Renormalize
        filtered_sum = filtered.sum(dim=-1, keepdim=True)
        result = filtered / filtered_sum

        return result


check_implementation = make_match_reference(ref_kernel, rtol=1e-3, atol=1e-5)
