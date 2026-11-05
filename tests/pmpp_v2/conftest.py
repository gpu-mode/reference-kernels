import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2] / "problems" / "pmpp_v2"
PROBLEMS = (
    "sort",
    "prefixsum",
    "histogram",
    "vectoradd",
    "vectorsum",
    "grayscale",
    "matmul",
    "conv2d",
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def load_problem():
    utils = load_module("pmpp_test_utils", ROOT / "utils.py")

    def load(name):
        folder = ROOT / f"{name}_py"
        task = load_module(f"pmpp_test_{name}_task", folder / "task.py")
        with patch.dict(sys.modules, {"task": task, "utils": utils}):
            return load_module(f"pmpp_test_{name}_reference", folder / "reference.py")

    return load


@pytest.fixture
def examples():
    gen = torch.Generator().manual_seed(72)
    return {
        "sort": (torch.tensor([3.0, -2.0, 1.0, 1.0, 0.0]), torch.empty(5)),
        "prefixsum": (torch.tensor([1e6, 1.0, -1e6, 2.0, -3.0, 4.0]), torch.empty(6)),
        "histogram": (
            torch.arange(256, dtype=torch.uint8).repeat(4),
            torch.empty(256, dtype=torch.int64),
        ),
        "vectoradd": (
            torch.randn(4, 4, dtype=torch.float16, generator=gen),
            torch.randn(4, 4, dtype=torch.float16, generator=gen),
            torch.empty(4, 4, dtype=torch.float16),
        ),
        "vectorsum": (torch.tensor([1e6, 1.0, -1e6, 2.0]), torch.empty(1)),
        "grayscale": (
            torch.tensor(
                [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]]
            ),
            torch.empty(2, 2),
        ),
        "matmul": (
            torch.randn(16, 16, dtype=torch.float16, generator=gen),
            torch.randn(16, 16, dtype=torch.float16, generator=gen),
            torch.empty(16, 16, dtype=torch.float16),
        ),
        "conv2d": (
            torch.randn(1, 2, 8, 8, generator=gen),
            torch.randn(2, 2, 4, 4, generator=gen),
            torch.empty(1, 2, 5, 5),
        ),
    }
