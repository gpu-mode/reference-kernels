import os
import sys
from pathlib import Path

if Path().resolve().name == "scripts":
    os.chdir("..")

sys.path.append("src/discord-cluster-manager")

from consts import ExitCode, SubmissionMode
from run_eval import run_pytorch_script

ref = Path("examples/identity_py/reference.py").read_text()
task = Path("examples/identity_py/task.py").read_text()
py_eval = Path("examples/eval.py").read_text()
utils = Path("examples/utils.py").read_text()
files = {"eval.py": py_eval, "reference.py": ref, "utils.py": utils, "task.py": task}


def run_pytorch_helper(sources: dict, **kwargs):
    result = run_pytorch_script(
        sources, "eval.py", mode=SubmissionMode.TEST.value, tests="size: 256; seed: 42\n", **kwargs
    )
    return result.run


def test_does_not_import():
    # input_tt is a typo, so this won't compile
    sub = """
    this is a syntax error
    """

    run = run_pytorch_helper({**files, "submission.py": sub})
    assert run.success is False
    assert run.exit_code != ExitCode.SUCCESS
    assert "IndentationError: unexpected indent\n" in run.stderr


def test_error():
    # no-op, runs fine but isn't correct
    sub = """
import torch
def custom_kernel(input):
    return torch.zeros_like(input)
        """

    run = run_pytorch_helper({**files, "submission.py": sub})
    assert run.success is True
    assert run.passed is False
    assert "python eval.py test" in run.command
    assert run.stdout == ""
    assert run.stderr == ""

    assert run.result["test.0.spec"] == "size: 256; seed: 42"
    assert run.result["test.0.status"] == "fail"
    assert (
        run.result["test.0.error"]
        == "mismatch found! custom implementation doesn't match reference.:"
        " Number of mismatched elements: 256"
    )
    assert run.exit_code == ExitCode.VALIDATE_FAIL
    assert run.result["check"] == "fail"


def test_correct():
    sub = Path("examples/identity_py/submission.py").read_text()

    run = run_pytorch_helper({**files, "submission.py": sub})
    assert run.success is True
    assert run.stdout == ""
    assert run.exit_code == ExitCode.SUCCESS
    assert run.result["check"] == "pass"


def test_huge_output():
    sub = """
import sys
def custom_kernel(input):
    print("blah blah\\n" * 10000, file=sys.stdout)
    return input
"""
    run = run_pytorch_helper({**files, "submission.py": sub})
    assert run.success
    assert len(run.stdout) < 16384
    assert "[...]" in run.stdout

    sub = sub.replace("sys.stdout", "sys.stderr")

    run = run_pytorch_helper({**files, "submission.py": sub})
    assert run.success
    assert len(run.stderr) < 16384
    assert "[...]" in run.stderr


def test_timeout():
    sub = """
from task import input_t, output_t
import time

def custom_kernel(data: input_t) -> output_t:
    time.sleep(5)
    return data
"""

    run = run_pytorch_helper({**files, "submission.py": sub}, test_timeout=2)
    assert run.success is False
    assert run.stdout == ""
    assert run.exit_code == ExitCode.TIMEOUT_EXPIRED
    assert len(run.result) == 0
