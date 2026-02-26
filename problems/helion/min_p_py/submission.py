from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    import torch

    logits, min_p = data
    logits = logits.float()

    probs = torch.softmax(logits, dim=-1)
    max_probs = probs.max(dim=-1, keepdim=True).values
    threshold = min_p[:, None] * max_probs
    filtered = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
    filtered_sum = filtered.sum(dim=-1, keepdim=True)
    result = filtered / filtered_sum
    return result
