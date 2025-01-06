import signal
import subprocess
from contextlib import contextmanager
from typing import Optional

from consts import MODAL_PATH
from modal import App, Image, Mount

# Create a stub for the Modal app
# IMPORTANT: This has to stay in separate file or modal breaks
mount = Mount.from_local_dir(
    MODAL_PATH,
    remote_path="/root/",
)
app = App("discord-bot-runner")
cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Move this to another file later:
python_image = Image.debian_slim(python_version="3.10").pip_install(["torch"])

cuda_image = (
    Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    # .apt_install(
    #     "git",
    #     "gcc-10",
    #     "g++-10",
    #     "clang",  # note i skip a step
    # )
    # .pip_install(  # required to build flash-attn
    #     "ninja",
    #     "packaging",
    #     "wheel",
    #     "torch",
    # )
    # .run_commands(
    #     # this is what we suppose to do but I am doing a shortcut
    #     # "update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100
    # --slave /usr/bin/g++ g++ /usr/bin/g++-10",
    #     # "apt update",
    #     # "apt  -y install clang-10", # this should be clang-10 but I can't get it to work yet
    #     #
    #     # "git clone https://github.com/HazyResearch/ThunderKittens.git",
    #     "git clone https://github.com/BradleyBrown19/ThunderMonkeys.git",  # TK + custom kernel
    #     force_build=True,  # always pull the latest
    #     # "cd /ThunderKittens && pwd && python setup.py install",
    # )
)


class TimeoutException(Exception):
    pass


@contextmanager
def timeout(seconds: int):
    """Context manager that raises TimeoutException after specified seconds"""

    def timeout_handler(signum, frame):
        raise TimeoutException(f"Script execution timed out after {seconds} seconds")

    # Set up the signal handler
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore the original handler and disable the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


@app.function(gpu="T4", image=python_image)
def run_pytorch_script(  # noqa: C901
    script_content: str,
    reference_content: Optional[str] = None,
    submission_content: Optional[str] = None,
    timeout_seconds: int = 300,
) -> tuple[str, float]:
    """
    Executes the provided PyTorch GPU kernel in an isolated environment with a timeout

    Args:
        script_content: The PyTorch script containing the GPU kernel to benchmark
        reference_content: The (optional) reference code, used for leaderboards.
        submission_content: The (optional) submission code, used for leaderboards.
        timeout_seconds: Maximum execution time before timeout (default: 300 seconds)

    Returns:
        tuple[str, float]: (Kernel output, execution time in milliseconds)

    NOTE: Modal execution time is not programmatically accessible, so we manually calculate it
    """

    import os
    import time

    try:
        with timeout(timeout_seconds):
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
                timeout=timeout_seconds,
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
                    return ("score", score)

            if score is None:
                execution_end_time = time.perf_counter()
                score = execution_end_time - execution_start_time

        return result.stdout, score

    except TimeoutException as e:
        return f"Timeout Error: {str(e)}", 0.0
    except Exception as e:
        return f"Error executing script: {str(e)}", 0.0
    finally:
        tmp_files = ["eval.py", "reference.py", "train.py"]
        for f in tmp_files:
            if os.path.exists(f):
                os.remove(f)


@app.function(
    gpu="T4",
    image=cuda_image,
)
def run_cuda_script(  # # noqa: C901
    script_content: str,
    reference_content: str = None,
    submission_content: str = None,
    timeout_seconds: int = 600,
) -> tuple[str, float]:
    """
    Executes the provided CUDA kernel in an isolated environment with a timeout

    Args:
        script_content: The CUDA script containing the GPU kernel
        reference_content: The (optional) reference code, used for leaderboards.
        submission_content: The (optional) submission code, used for leaderboards.
        timeout_seconds: Maximum execution time in seconds (default: 600 seconds)

    Returns:
        tuple[str, float]: (Kernel output, execution time in milliseconds)

    NOTE: Modal execution time is not programmatically accessible, so we manually calculate it
    """
    import os
    import subprocess
    import time

    try:
        with timeout(timeout_seconds):
            # Check CUDA is available and installed correctly
            print("[CUDA Env Check]")
            try:
                # these check cuda compiler is also available
                subprocess.run(["nvcc", "--version"], check=True)
                subprocess.run(["which", "nvcc"], check=True)
            except Exception:
                return "nvcc not found.", 0.0

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
                ["nvcc", "--std=c++17", NVCC_FILES, "-o", "eval.out"],
                capture_output=True,
                text=True,
            )
            compilation_output = compile_process.stdout
            compilation_error = compile_process.stderr
            print("out", compilation_output)
            print("err", compilation_error)

            print("return code", compile_process.returncode)
            if compile_process.returncode != 0:
                raise RuntimeError(
                    "CUDA compilation failed with return code "
                    + f"{compile_process.returncode}:\n{compile_process.stderr}"
                )

            run_process = subprocess.run(["./eval.out"], capture_output=True, text=True)
            execution_end_time = time.perf_counter()

            score = None
            for line in run_process.stdout.splitlines():
                if line.startswith("score:"):
                    score = float(line.split(":")[1].strip())
                    return ("score", score)

            if score is None:
                execution_end_time = time.perf_counter()
                score = execution_end_time - execution_start_time

            return run_process.stdout, score

    except TimeoutException as e:
        return f"Timeout Error: {str(e)}", 0.0
    except Exception as e:
        return f"Error: {str(e)}", 0.0
    finally:
        tmp_files = ["reference.cuh", "train.cuh", "eval.cu", "eval.out"]
        for f in tmp_files:
            if os.path.exists(f):
                os.remove(f)
