import os
import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(use_cuda: bool = True) -> torch.device:
    """Get the appropriate device (GPU or CPU)."""
    if use_cuda:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("No compatible GPU found. Falling back to CPU.")
    return torch.device("cpu")


def _check_tensor_contract(received, expected) -> list[str]:
    """Reject malformed results before comparisons can broadcast or promote them."""
    if not isinstance(received, torch.Tensor):
        return ["TYPE MISMATCH: output must be a torch.Tensor"]
    if received.shape != expected.shape:
        return [
            f"SIZE MISMATCH: got {tuple(received.shape)}, expected {tuple(expected.shape)}"
        ]
    if received.dtype != expected.dtype:
        return [f"DTYPE MISMATCH: got {received.dtype}, expected {expected.dtype}"]
    if received.device != expected.device:
        return [f"DEVICE MISMATCH: got {received.device}, expected {expected.device}"]
    if expected.is_floating_point():
        if not torch.isfinite(expected).all().item():
            return ["NON-FINITE REFERENCE: PMPP v2 requires finite results"]
        if not torch.isfinite(received).all().item():
            return ["NON-FINITE OUTPUT"]
    return []


def _mismatch_details(received, expected, mismatched, max_print):
    num_mismatched = mismatched.count_nonzero().item()
    if not num_mismatched:
        return []
    details = [f"Number of mismatched elements: {num_mismatched}"]
    # A fabricated large result can mismatch everywhere. Never allocate an
    # index tensor proportional to all failures just to report five examples.
    flat = mismatched.reshape(-1)
    printed = 0
    for start in range(0, flat.numel(), 65536):
        indices = flat[start : start + 65536].nonzero().flatten()[: max_print - printed]
        for offset in indices.tolist():
            position = start + offset
            index = []
            for size in reversed(expected.shape):
                index.append(position % size)
                position //= size
            index = tuple(reversed(index))
            details.append(f"ERROR AT {index}: {received[index]} {expected[index]}")
            printed += 1
        if printed >= max_print:
            break
    if num_mismatched > printed:
        details.append(f"... and {num_mismatched - printed} more mismatched elements.")
    return details


# Adapted from https://github.com/linkedin/Liger-Kernel/blob/main/test/utils.py
@torch.no_grad()
def verbose_allclose(
    received: torch.Tensor, expected: torch.Tensor, rtol=1e-05, atol=1e-08, max_print=5
) -> list[str]:
    """
    Assert that two tensors are element-wise equal within a tolerance, providing detailed information about mismatches.

    Parameters:
    received (torch.Tensor): Tensor we actually got.
    expected (torch.Tensor): Tensor we expected to receive.
    rtol (float): Relative tolerance; relative to expected
    atol (float): Absolute tolerance.
    max_print (int): Maximum number of mismatched elements to print.

    Raises:
    AssertionError: If the tensors are not all close within the given tolerance.
    """
    reasons = _check_tensor_contract(received, expected)
    if reasons:
        return reasons

    # Calculate the difference between the tensors
    diff = torch.abs(received - expected)

    # Determine the tolerance
    tolerance = atol + rtol * torch.abs(expected)

    # Non-finite values were rejected above. Keep only one mismatch mask for
    # benchmark-sized scans instead of several full-size NaN/Inf masks.
    mismatched = diff > tolerance

    return _mismatch_details(received, expected, mismatched, max_print)


@torch.no_grad()
def verbose_allequal(
    received: torch.Tensor, expected: torch.Tensor, max_print: int = 5
):
    """
    Assert that two tensors are element-wise perfectly equal, providing detailed information about mismatches.

    Parameters:
    received (torch.Tensor): Tensor we actually got.
    expected (torch.Tensor): Tensor we expected to receive.
    max_print (int): Maximum number of mismatched elements to print.

    Returns:
         Empty string if tensors are equal, otherwise detailed error information
    """
    reasons = _check_tensor_contract(received, expected)
    if reasons:
        return reasons
    mismatched = torch.not_equal(received, expected)
    return _mismatch_details(received, expected, mismatched, max_print)


def match_reference(
    data, output, reference: callable, rtol=1e-05, atol=1e-08
) -> tuple[bool, str]:
    """
    Convenient "default" implementation for tasks' `check_implementation` function.
    """
    expected = reference(data)
    reasons = verbose_allclose(output, expected, rtol=rtol, atol=atol)

    if len(reasons) > 0:
        return (
            False,
            "mismatch found! custom implementation doesn't match reference: "
            + " ".join(reasons),
        )

    return True, ""


def make_match_reference(reference: callable, **kwargs):
    def wrapped(data, output):
        return match_reference(data, output, reference=reference, **kwargs)

    return wrapped


def make_match_reference_exact(reference: callable):
    """Use for operations such as sorting that must preserve exact values."""

    @torch.no_grad()
    def wrapped(data, output):
        expected = reference(data)
        reasons = verbose_allequal(output, expected)
        if reasons:
            return False, "exact reference mismatch: " + " ".join(reasons)
        return True, ""

    return wrapped


class DeterministicContext:
    def __init__(self):
        self.allow_tf32 = None
        self.deterministic = None
        self.cublas = None
        self.algorithms = None
        self.warn_only = None
        self.matmul_tf32 = None

    def __enter__(self):
        self.cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
        self.allow_tf32 = torch.backends.cudnn.allow_tf32
        self.deterministic = torch.backends.cudnn.deterministic
        self.algorithms = torch.are_deterministic_algorithms_enabled()
        self.warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        self.matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.backends.cudnn.allow_tf32 = self.allow_tf32
        torch.backends.cudnn.deterministic = self.deterministic
        torch.backends.cuda.matmul.allow_tf32 = self.matmul_tf32
        torch.use_deterministic_algorithms(self.algorithms, warn_only=self.warn_only)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = self.cublas


def clear_l2_cache():
    # import cupy as cp
    # cp.cuda.runtime.deviceSetLimit(cp.cuda.runtime.cudaLimitPersistingL2CacheSize, 0)
    # create a large dummy tensor
    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda")
    # write stuff to
    dummy.fill_(42)
    del dummy
