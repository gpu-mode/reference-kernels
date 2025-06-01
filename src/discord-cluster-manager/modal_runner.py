import signal
import traceback
from contextlib import contextmanager

from modal import App, Image
from run_eval import FullResult, SystemInfo, run_config

# Create a stub for the Modal app
# IMPORTANT: This has to stay in separate file or modal breaks
app = App("discord-bot-runner")
cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Move this to another file later:
cuda_image = (
    Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install(
        "git",
        "gcc-11",
        "g++-11",
        "clang-11",  # note i skip a step
    )
    .pip_install(
        "ninja",
        "packaging",
        "wheel",
        "torch",
        "numpy",
        "triton",
        "jax[cuda12]",
        "jax2torch",
        "tinygrad",
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
    .pip_install("requests")
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
