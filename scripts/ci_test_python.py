import os
import sys
from pathlib import Path

if Path().resolve().name == "scripts":
    os.chdir("..")

sys.path.append("src/discord-cluster-manager")

from consts import ExitCode
from leaderboard_eval import py_eval
from run_eval import run_pytorch_script

ref = Path("examples/identity_py/reference.py")


def test_does_not_import():
    # input_tt is a typo, so this won't compile
    sub = """
    this is a syntax error
    """

    run = run_pytorch_script(
        {"eval.py": py_eval, "reference.py": ref.read_text(), "submission.py": sub}, "eval.py"
    )
    assert run.success is False
    assert run.exit_code != ExitCode.SUCCESS
    assert "IndentationError: unexpected indent\n" in run.stderr


def test_error():
    # no-op, runs fine but isn't correct
    sub = """
import torch
def custom_kernel(input):
    return [torch.zeros_like(i) for i in input]
        """

    run = run_pytorch_script(
        {"eval.py": py_eval, "reference.py": ref.read_text(), "submission.py": sub},
        "eval.py",
        arch=None,
    )
    assert run.success is True
    assert run.passed is False
    assert run.command == "python eval.py"
    # we never reach the benchmark part, because the test fails
    assert "warming up..." not in run.stdout
    assert "mismatch found! custom implementation doesnt match reference." in run.stdout
    assert run.exit_code == ExitCode.VALIDATE_FAIL
    assert run.result["check"] == "fail"


def test_correct():
    sub = Path("examples/identity_py/submission.py").read_text()

    run = run_pytorch_script(
        {"eval.py": py_eval, "reference.py": ref.read_text(), "submission.py": sub}, "eval.py"
    )
    assert run.success is True
    assert "warming up..." in run.stdout
    assert run.exit_code == ExitCode.SUCCESS
    assert run.result["check"] == "pass"
