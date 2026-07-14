import sys
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch


def _load_eval_module():
    reference = types.ModuleType("reference")
    reference.check_implementation = lambda data, output: (True, "")
    reference.generate_input = lambda **kwargs: kwargs

    task = types.ModuleType("task")
    task.TestSpec = dict

    utils = types.ModuleType("utils")
    utils.clear_l2_cache = lambda: None
    utils.set_seed = lambda seed: None

    name = "eigh_eval_under_test"
    path = Path(__file__).with_name("eval.py")
    source = "from __future__ import annotations\n" + path.read_text()
    module = types.ModuleType(name)
    module.__file__ = str(path)
    with patch.dict(
        sys.modules,
        {name: module, "reference": reference, "task": task, "utils": utils},
    ):
        exec(compile(source, str(path), "exec"), module.__dict__)
    return module


class _FakeEvent:
    def __init__(self, enable_timing=False):
        self.enable_timing = enable_timing

    def record(self):
        pass

    def elapsed_time(self, other):
        return 1.0


class FreshBenchmarkInputTest(unittest.TestCase):
    def test_content_memoization_never_hits_warmup_or_previous_runs(self):
        evaluator = _load_eval_module()
        generated_seeds = []
        cached_outputs = {}
        cache_hits = 0

        def generate_input(**args):
            generated_seeds.append(args["seed"])
            return torch.tensor([args["seed"]], dtype=torch.int64)

        def custom_kernel(data):
            nonlocal cache_hits
            key = int(data.item())
            if key in cached_outputs:
                cache_hits += 1
                return cached_outputs[key].clone()
            output = data.clone()
            cached_outputs[key] = output.clone()
            return output

        def check_implementation(data, output):
            return torch.equal(data, output), "memoized output used for different input"

        submission = types.ModuleType("submission")
        submission.custom_kernel = custom_kernel
        test = evaluator.TestCase(
            args={"batch": 1, "n": 1, "cond": 0, "seed": 1234},
            spec="unit-test",
        )

        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"submission": submission}))
            stack.enter_context(patch.object(evaluator, "generate_input", generate_input))
            stack.enter_context(
                patch.object(evaluator, "check_implementation", check_implementation)
            )
            stack.enter_context(
                patch.object(evaluator, "_benchmark_batch_count", return_value=1)
            )
            stack.enter_context(patch.object(evaluator, "clear_l2_cache", return_value=None))
            stack.enter_context(
                patch.object(evaluator.torch.cuda, "synchronize", return_value=None)
            )
            stack.enter_context(patch.object(evaluator.torch.cuda, "Event", _FakeEvent))
            stack.enter_context(
                patch.object(evaluator.time, "perf_counter_ns", return_value=0)
            )
            stack.enter_context(
                patch.object(
                    evaluator,
                    "secure_token_bytes",
                    side_effect=(bytes([1]) * 16, bytes([2]) * 16),
                )
            )
            first = evaluator._run_single_benchmark(test, True, 4, float("inf"))
            second = evaluator._run_single_benchmark(test, True, 4, float("inf"))

        self.assertIsInstance(first, evaluator.Stats)
        self.assertIsInstance(second, evaluator.Stats)
        self.assertEqual(first.runs, 4)
        self.assertEqual(second.runs, 4)
        self.assertEqual(cache_hits, 0)
        self.assertEqual(len(generated_seeds), 10)
        self.assertEqual(len(set(generated_seeds)), len(generated_seeds))


if __name__ == "__main__":
    unittest.main()
