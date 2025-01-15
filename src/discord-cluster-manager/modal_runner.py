import signal
from contextlib import contextmanager
from typing import Optional

from consts import MODAL_CUDA_INCLUDE_DIRS, MODAL_PATH
from modal import App, Image, Mount
from run_eval import FullResult, run_cuda_script, run_pytorch_script

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
    [
        "torch",
        "triton",
        "jax[cuda12]",
        "jax2torch",
    ]
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
) -> FullResult:
    """Modal version of run_pytorch_script, handling timeouts"""
    try:
        with timeout(timeout_seconds):
            run_result = run_pytorch_script(
                {
                    "eval.py": script_content,
                    "reference.py": reference_content,
                    "submission.py": submission_content,
                },
                "eval.py",
            )
            return FullResult(success=True, error="", compile=None, run=run_result)
        # TODO fixup error handling!
    except TimeoutException as e:
        return FullResult(success=False, error=f"Timeout Error: {str(e)}", compile=None, run=None)
    except Exception as e:
        return FullResult(
            success=False, error=f"Error executing script: {str(e)}", compile=None, run=None
        )


def modal_run_cuda_script(  # # noqa: C901
    script_content: str,
    reference_content: str = None,
    submission_content: str = None,
    timeout_seconds: int = 600,
    arch: int = None,
) -> FullResult:
    """Modal version of run_cuda_script, handling timeouts"""
    try:
        with timeout(timeout_seconds):
            comp, run = run_cuda_script(
                {"eval.cu": script_content},
                {"reference.cuh": reference_content, "submission.cuh": submission_content},
                arch=arch,
                include_dirs=MODAL_CUDA_INCLUDE_DIRS,
            )
        return FullResult(success=True, error="", compile=comp, run=run)
    # TODO fixup error handling!
    except TimeoutException as e:
        return FullResult(success=False, error=f"Timeout Error: {str(e)}", compile=None, run=None)
    except Exception as e:
        return FullResult(
            success=False, error=f"Error executing script: {str(e)}", compile=None, run=None
        )
