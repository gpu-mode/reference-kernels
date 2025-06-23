import signal
import traceback
from contextlib import contextmanager

from modal import App, Image
from run_eval import FullResult, SystemInfo, run_config

# Create a stub for the Modal app
# IMPORTANT: This has to stay in separate file or modal breaks
app = App("discord-bot-runner")
cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Move this to another file later:
cuda_image = (
    Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install(
        "git",
        "gcc-13",
        "g++-13",
        "clang-18",
    )
    .pip_install(
        "ninja~=1.11",
        "wheel~=0.45",
        "requests~=2.32.4",
        "packaging~=25.0",
        "numpy~=2.3",
    )
    .pip_install(
        "torch~=2.7",
        "torchvision~=0.22",
        "torchaudio~=2.7",
        index_url="https://download.pytorch.org/whl/cu128"
    )
    # other frameworks
    .pip_install(
        "jax[cuda12]==0.5.3",   # 0.6 want's cudnn 9.8 in conflict with torch 2.7
        "jax2torch==0.0.7",
        "tinygrad~=0.10",
    )
    # nvidia cuda packages
    .pip_install(
        "nvidia-cupynumeric~=25.3",
        "nvidia-cutlass-dsl~=4.0",
        "cuda-core[cu12]~=0.3",
        "cuda-python[all]==12.8",
        #"nvmath-python[cu12]~=0.4",
        #"numba-cuda[cu12]~=0.15",
    )
)

cuda_image = cuda_image.add_local_python_source(
    "consts",
    "modal_runner",
    "modal_runner_archs",
    "run_eval",
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


def modal_run_config(  # noqa: C901
    config: dict,
    timeout_seconds: int = 300,
) -> FullResult:
    """Modal version of run_pytorch_script, handling timeouts"""
    try:
        with timeout(timeout_seconds):
            return run_config(config)
    except TimeoutException as e:
        return FullResult(
            success=False,
            error=f"Timeout Error: {str(e)}",
            runs={},
            system=SystemInfo(),
        )
    except Exception as e:
        exception = "".join(traceback.format_exception(e))
        return FullResult(
            success=False,
            error=f"Error executing script:\n{exception}",
            runs={},
            system=SystemInfo(),
        )
