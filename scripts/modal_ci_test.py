#!/usr/bin/env python3
"""
Modal CI test runner - runs key test scenarios on Modal
"""
import os
import sys
from pathlib import Path

import modal

# Change to the correct directory
if os.path.basename(os.getcwd()) == "scripts":
    os.chdir("..")

# Add the src directory to Python path for Modal deserialization
sys.path.append("src/discord-cluster-manager")

from consts import SubmissionMode
from task import build_task_config, make_task_definition


def test_cuda_correct():
    """Test that correct CUDA submission passes"""
    print("Testing CUDA correct submission...")

    func = modal.Function.from_name("discord-bot-runner", "run_cuda_script_t4")
    task = make_task_definition("examples/identity_cuda")
    submission_cu = Path("examples/identity_cuda/submission.cu").read_text()

    config = build_task_config(
        task=task.task,
        submission_content=submission_cu,
        arch=None,
        mode=SubmissionMode.TEST,
    )

    result = func.remote(config=config)

    if not result.success:
        raise Exception(f"CUDA test failed: {result.error}")

    # Check if any test runs failed
    for run_name, run_result in result.runs.items():
        if run_result.run and not run_result.run.success:
            raise Exception(f"Test run {run_name} failed")

    print("✅ CUDA correct submission passed")


def test_cuda_validation_fail():
    """Test that incorrect CUDA submission fails validation"""
    print("Testing CUDA validation failure...")

    func = modal.Function.from_name("discord-bot-runner", "run_cuda_script_t4")
    task = make_task_definition("examples/identity_cuda")

    # no-op submission that should fail validation
    submission_cu = """
    #include "task.h"

    output_t custom_kernel(input_t data)
    {
        output_t result;
        result.resize(data.size());
        return result;
    }
    """

    config = build_task_config(
        task=task.task,
        submission_content=submission_cu,
        arch=None,
        mode=SubmissionMode.TEST,
    )

    result = func.remote(config=config)

    if not result.success:
        raise Exception(f"CUDA test failed to execute: {result.error}")

    # Should have a test run that fails validation
    test_run = result.runs.get("test")
    if not test_run or not test_run.run:
        raise Exception("No test run found")

    if test_run.run.passed:
        raise Exception("Expected validation failure but test passed")

    print("✅ CUDA validation failure test passed")


def test_pytorch_correct():
    """Test that correct PyTorch submission passes"""
    print("Testing PyTorch correct submission...")

    func = modal.Function.from_name("discord-bot-runner", "run_pytorch_script_t4")
    task = make_task_definition("examples/identity_py")
    submission_py = Path("examples/identity_py/submission.py").read_text()

    config = build_task_config(
        task=task.task,
        submission_content=submission_py,
        arch=None,
        mode=SubmissionMode.TEST,
    )

    result = func.remote(config=config)

    if not result.success:
        raise Exception(f"PyTorch test failed: {result.error}")

    # Check if any test runs failed
    for run_name, run_result in result.runs.items():
        if run_result.run and not run_result.run.success:
            raise Exception(f"Test run {run_name} failed")

    print("✅ PyTorch correct submission passed")


def test_pytorch_validation_fail():
    """Test that incorrect PyTorch submission fails validation"""
    print("Testing PyTorch validation failure...")

    func = modal.Function.from_name("discord-bot-runner", "run_pytorch_script_t4")
    task = make_task_definition("examples/identity_py")

    # no-op submission that should fail validation
    submission_py = """
import torch
def custom_kernel(input):
    return torch.zeros_like(input)
    """

    config = build_task_config(
        task=task.task,
        submission_content=submission_py,
        arch=None,
        mode=SubmissionMode.TEST,
    )

    result = func.remote(config=config)

    if not result.success:
        raise Exception(f"PyTorch test failed to execute: {result.error}")

    # Should have a test run that fails validation
    test_run = result.runs.get("test")
    if not test_run or not test_run.run:
        raise Exception("No test run found")

    if test_run.run.passed:
        raise Exception("Expected validation failure but test passed")

    print("✅ PyTorch validation failure test passed")


def main():
    """Run all Modal tests"""
    print("Running Modal CI tests...")

    try:
        test_cuda_correct()
        test_cuda_validation_fail()
        test_pytorch_correct()
        test_pytorch_validation_fail()

        print("\n🎉 All Modal tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
