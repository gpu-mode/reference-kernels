from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    import torch

    logits, top_k = data
    logits = logits.float()
    batch_size, vocab_size = logits.shape

    probs = torch.softmax(logits, dim=-1)
    filtered = torch.zeros_like(probs)
    for b in range(batch_size):
        k = min(top_k[b].item(), vocab_size)
        topk_vals, _ = torch.topk(probs[b], k)
        threshold = topk_vals[-1]
        mask = probs[b] >= threshold
        filtered[b] = torch.where(mask, probs[b], torch.zeros_like(probs[b]))

    filtered_sum = filtered.sum(dim=-1, keepdim=True)
    result = filtered / filtered_sum
    return result
