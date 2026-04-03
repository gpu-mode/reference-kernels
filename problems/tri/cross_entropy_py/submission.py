"""
Baseline submission for the cross-entropy problem.

Replace these functions with a faster implementation.

Example local bandwidth calculation for the three ranked shapes:

    def print_max_bw(batch_size, vocab_size, combined_ms):
        total_bytes = (6 * batch_size * vocab_size) + (24 * batch_size)
        combined_bw = total_bytes / (combined_ms * 1e-3) / 1e9
        print(f\"B={batch_size} V={vocab_size}: {combined_bw:.2f} GB/s\")

This is only for local debugging. Do not add timing calls inside the hot path
if you care about leaderboard performance.
"""

import torch
import torch.nn.functional as F


def cross_entropy_forward(logits, targets):
    """
    Args:
        logits: (B, V) torch.bfloat16
        targets: (B,) torch.int64
    Returns:
        (B,) torch.float32
    """
    return F.cross_entropy(logits.float(), targets, reduction="none")


def cross_entropy_backward(logits, targets, grad_output):
    """
    Args:
        logits: (B, V) torch.bfloat16
        targets: (B,) torch.int64
        grad_output: (B,) torch.float32
    Returns:
        (B, V) torch.bfloat16
    """
    probs = torch.softmax(logits.float(), dim=-1)
    probs[torch.arange(logits.shape[0], device=logits.device), targets] -= 1.0
    grad_logits = probs * grad_output.unsqueeze(1)
    return grad_logits.to(logits.dtype)
