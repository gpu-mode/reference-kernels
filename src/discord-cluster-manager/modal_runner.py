import signal
from contextlib import contextmanager
from typing import Optional

from consts import MODAL_CUDA_INCLUDE_DIRS, MODAL_PATH
from modal import App, Image, Mount
from run_eval import run_cuda_script, run_pytorch_script

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
python_image = Image.debian_slim(python_version="3.10").pip_install(
    ["torch", "triton", "jax[cuda12]", "jax2torch"]
)

cuda_image = (
    Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install(
        "git",
        "gcc-11",
        "g++-11",
        "clang-11",  # note i skip a step
    )
    .pip_install(
        "ninja", "packaging", "wheel", "torch", "numpy", "triton", "jax[cuda12]", "jax2torch"
    )
    .run_commands(
        "update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 "
        + "--slave /usr/bin/g++ g++ /usr/bin/g++-11",
        # "apt update",
        # "apt  -y install clang-10", # this should be clang-10 but I can't get it to work yet
        #
        "git clone https://github.com/HazyResearch/ThunderKittens.git",
        # "cd /ThunderKittens && pwd && python setup.py install",
    )
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


def modal_run_pytorch_script(  # noqa: C901
    script_content: str,
    reference_content: Optional[str] = None,
    submission_content: Optional[str] = None,
    timeout_seconds: int = 300,
    arch: int = None,
) -> tuple[str, float]:
    """Modal version of run_pytorch_script, handling timeouts"""
    try:
        with timeout(timeout_seconds):
            return run_pytorch_script(
                script_content=script_content,
                reference_content=reference_content,
                submission_content=submission_content,
                arch=arch,
            )

    except TimeoutException as e:
        return f"Timeout Error: {str(e)}", 0.0


def modal_run_cuda_script(  # # noqa: C901
    script_content: str,
    reference_content: str = None,
    submission_content: str = None,
    timeout_seconds: int = 600,
    arch: int = None,
) -> tuple[str, float]:
    """Modal version of run_cuda_script, handling timeouts"""
    try:
        with timeout(timeout_seconds):
            compile_result, run_result = run_cuda_script(
                script_content,
                reference_content=reference_content,
                submission_content=submission_content,
                arch=arch,
                include_dirs=MODAL_CUDA_INCLUDE_DIRS,
            )

            if not compile_result.success:
                if not compile_result.nvcc_found:
                    return (
                        "Error executing script: NVCC not found:\n"
                        + f"command `{compile_result.command}` "
                        + f"failed with exit code {compile_result.exit_code}:\n"
                        + compile_result.stderr,
                        0.0,
                    )
                return (
                    "Error executing script: CUDA compilation failed with return code "
                    + f"{compile_result.exit_code}:\n{compile_result.stderr}\n"
                    + f"compile command: `{compile_result.command}`",
                    0.0,
                )

            if not run_result.success:
                # exit code 1 encodes failed tests
                if run_result.exit_code == 1:
                    return f"check_implementation failed:\n{run_result.stderr}", 0.0
                else:
                    return (
                        f"Script failed with exit code "
                        f"({run_result.exit_code}):\n{run_result.stderr}",
                        0.0,
                    )

            print("run process stdout:", run_result.stdout)
            print("run process stderr:", run_result.stderr)

            score = float(run_result.result.get("duration.mean", "0.0")) / 1e9
            passed = run_result.result.get("check", "") == "pass"
            if not passed:
                return "check_implementation failed", 0.0

            if score is None:
                return run_result.stdout, run_result.duration

            return run_result.stdout, score

    except TimeoutException as e:
        return f"Timeout Error: {str(e)}", 0.0
    except Exception as e:
        return f"Error executing script: {str(e)}", 0.0
