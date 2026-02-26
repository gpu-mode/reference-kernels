from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    import torch

    logits, top_p = data
    logits = logits.float()

    probs = torch.softmax(logits, dim=-1)
    sorted_probs, _ = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    shifted_cumsum = cumsum - sorted_probs
    nucleus_mask = shifted_cumsum <= top_p[:, None]
    masked_sorted = torch.where(nucleus_mask, sorted_probs, torch.ones_like(sorted_probs))
    threshold = masked_sorted.amin(dim=-1, keepdim=True)
    filtered = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
    filtered_sum = filtered.sum(dim=-1, keepdim=True)
    result = filtered / filtered_sum
    return result
