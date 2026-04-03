import torch
import torch.nn.functional as F


DTYPE = torch.bfloat16
DEVICE = "cuda"
ATOL = 1e-3
RTOL = 1e-2


def reference_forward(logits, targets):
    return F.cross_entropy(logits.float(), targets, reduction="none")


def reference_backward(logits, targets, grad_output):
    probs = torch.softmax(logits.float(), dim=-1)
    grad = probs
    grad[torch.arange(logits.shape[0], device=logits.device), targets] -= 1.0
    grad = grad * grad_output.unsqueeze(1)
    return grad.to(logits.dtype)


def generate_inputs(batch_size, vocab_size, seed=42):
    torch.manual_seed(seed)
    logits = torch.randn(batch_size, vocab_size, dtype=DTYPE, device=DEVICE)
    targets = torch.randint(0, vocab_size, (batch_size,), device=DEVICE)
    grad_output = torch.randn(batch_size, dtype=torch.float32, device=DEVICE)
    return logits, targets, grad_output
