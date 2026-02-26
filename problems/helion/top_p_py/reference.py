import torch
from task import input_t, output_t
from utils import make_match_reference, DeterministicContext


def generate_input(batch_size: int, vocab_size: int, seed: int) -> input_t:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    logits = torch.randn(batch_size, vocab_size, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    # top_p values between 0.85 and 0.95
    top_p = (torch.rand(batch_size, device="cuda", generator=gen) * 0.1 + 0.85).to(torch.float32).contiguous()
    return logits, top_p


def ref_kernel(data: input_t) -> output_t:
    with DeterministicContext():
        logits, top_p = data
        logits = logits.float()

        # 1. Softmax
        probs = torch.softmax(logits, dim=-1)

        # 2. Sort descending
        sorted_probs, _sorted_indices = torch.sort(probs, descending=True, dim=-1)

        # 3. Cumulative sum
        cumsum = torch.cumsum(sorted_probs, dim=-1)

        # 4. Find threshold per batch element
        shifted_cumsum = cumsum - sorted_probs
        nucleus_mask = shifted_cumsum <= top_p[:, None]
        masked_sorted = torch.where(nucleus_mask, sorted_probs, torch.ones_like(sorted_probs))
        threshold = masked_sorted.amin(dim=-1, keepdim=True)

        # 5. Apply threshold to original probs
        filtered = torch.where(probs >= threshold, probs, torch.zeros_like(probs))

        # 6. Renormalize
        filtered_sum = filtered.sum(dim=-1, keepdim=True)
        result = filtered / filtered_sum

        return result


check_implementation = make_match_reference(ref_kernel, rtol=1e-3, atol=1e-5)
