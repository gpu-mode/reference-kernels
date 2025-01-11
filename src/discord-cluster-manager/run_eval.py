import os
import subprocess
import time
from typing import Optional

from consts import CUDA_FLAGS


def run_cuda_script(  # # noqa: C901
    script_content: str,
    reference_content: str = None,
    submission_content: str = None,
    arch: int = None,
    include_dirs: list[str] = None,
) -> tuple[str, float]:
    """
    Executes the provided CUDA kernel in an isolated environment with a timeout

    Args:
        script_content: The CUDA script containing the GPU kernel
        reference_content: The (optional) reference code, used for leaderboards.
        submission_content: The (optional) submission code, used for leaderboards.
        arch: The arch code for the compute/sm versions. If None, native arch is used.
        include_dirs: Additional include directories, e.g., for thunderkittens/cutlass etc

    Returns:
        tuple[str, float]: (Kernel output, execution time in milliseconds)
    """
    if include_dirs is None:
        include_dirs = []

    try:
        # Check CUDA is available and installed correctly
        print("[CUDA Env Check]")
        try:
            # these check cuda compiler is also available
            print(subprocess.check_output(["which", "nvcc"], encoding="utf-8"))
            print(subprocess.check_output(["nvcc", "--version"], encoding="utf-8"))
        except Exception:
            return "nvcc not found.", 0.0

        if arch is None:
            ARCH = "-arch=native"
        else:
            ARCH = f"-gencode=arch=compute_{arch},code=sm_{arch}"
        NVCC_FILES = "eval.cu"
        # Write submission files to directory
        if reference_content is not None:
            with open("reference.cuh", "w") as f:
                f.write(reference_content)

        if submission_content is not None:
            with open("train.cuh", "w") as f:
                f.write(submission_content)

        with open("eval.cu", "w") as f:
            f.write(script_content)

        execution_start_time = time.perf_counter()
        compile_process = subprocess.run(
            ["nvcc"] + CUDA_FLAGS + include_dirs + [ARCH, NVCC_FILES, "-o", "eval.out"],
            capture_output=True,
            text=True,
        )

        if compile_process.returncode != 0:
            raise RuntimeError(
                "CUDA compilation failed with return code "
                + f"{compile_process.returncode}:\n{compile_process.stderr}"
            )

        run_process = subprocess.run(["./eval.out"], capture_output=True, text=True)
        execution_end_time = time.perf_counter()

        print("run process stdout", run_process.stdout)
        print("run process stderr", run_process.stderr)

        score = None
        for line in run_process.stdout.splitlines():
            if line.startswith("score:"):
                score = float(line.split(":")[1].strip())
                break

        if score is None:
            execution_end_time = time.perf_counter()
            score = execution_end_time - execution_start_time
            if "check_implementation failed" in run_process.stdout:
                return "check_implementation failed", 0.0
            else:
                return None, score

        return run_process.stdout, score

    except Exception as e:
        return f"Error executing script: {str(e)}", 0.0
    finally:
        tmp_files = ["reference.cuh", "train.cuh", "eval.cu", "eval.out"]
        for f in tmp_files:
            if os.path.exists(f):
                os.remove(f)


def run_pytorch_script(  # noqa: C901
    script_content: str,
    reference_content: Optional[str] = None,
    submission_content: Optional[str] = None,
    arch: int = None,
) -> tuple[str, float]:
    """
    Executes the provided PyTorch GPU kernel in an isolated environment with a timeout

    Args:
        script_content: The PyTorch script containing the GPU kernel to benchmark
        reference_content: The (optional) reference code, used for leaderboards.
        submission_content: The (optional) submission code, used for leaderboards.
        arch: The arch code for the compute/sm versions.

    Returns:
        tuple[str, float]: (Kernel output, execution time in milliseconds)
    """
    try:
        # Write submission files to directory
        if reference_content is not None:
            with open("reference.py", "w") as f:
                f.write(reference_content)

        if submission_content is not None:
            with open("train.py", "w") as f:
                f.write(submission_content)

        with open("eval.py", "w") as f:
            f.write(script_content)

        execution_start_time = time.perf_counter()
        result = subprocess.run(
            ["python", "eval.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Script execution failed with return code "
                + f"{result.returncode}:\n{result.stderr}"
            )

        score = None
        for line in result.stdout.splitlines():
            if line.startswith("score:"):
                score = float(line.split(":")[1].strip())
                return "score", score

        if score is None:
            execution_end_time = time.perf_counter()
            score = execution_end_time - execution_start_time

        return result.stdout, score
    except Exception as e:
        return f"Error executing script: {str(e)}", 0.0
    finally:
        tmp_files = ["eval.py", "reference.py", "train.py"]
        for f in tmp_files:
            if os.path.exists(f):
                os.remove(f)
