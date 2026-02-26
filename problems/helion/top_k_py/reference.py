import torch
from task import input_t, output_t
from utils import make_match_reference, DeterministicContext


def generate_input(batch_size: int, vocab_size: int, k: int, seed: int) -> input_t:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    logits = torch.randn(batch_size, vocab_size, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    top_k = torch.full((batch_size,), k, dtype=torch.int32, device="cuda")
    return logits, top_k


def ref_kernel(data: input_t) -> output_t:
    with DeterministicContext():
        logits, top_k = data
        logits = logits.float()
        batch_size, vocab_size = logits.shape

        # 1. Softmax
        probs = torch.softmax(logits, dim=-1)

        # 2. Top-k filtering (per-batch k)
        filtered = torch.zeros_like(probs)
        for b in range(batch_size):
            k = min(top_k[b].item(), vocab_size)
            topk_vals, _ = torch.topk(probs[b], k)
            threshold = topk_vals[-1]
            mask = probs[b] >= threshold
            filtered[b] = torch.where(mask, probs[b], torch.zeros_like(probs[b]))

        # 3. Renormalize
        filtered_sum = filtered.sum(dim=-1, keepdim=True)
        result = filtered / filtered_sum

        return result


check_implementation = make_match_reference(ref_kernel, rtol=1e-3, atol=1e-5)
