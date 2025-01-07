# This file contains wrapper functions for running
# Modal apps on specific devices. We will fix this later.

import sys
from contextlib import contextmanager

from consts import GPU_TO_SM
from modal_runner import app, cuda_image, run_cuda_script, run_pytorch_script


# T4: sm_70 (CUDA 7.x, Maxwell Architecture)
@app.function(
    gpu="T4",
    image=cuda_image,
)
def run_cuda_script_t4(
    script_content: str,
    reference_content: str = None,
    submission_content: str = None,
    timeout_seconds: int = 600,
) -> tuple[str, float]:
    return run_cuda_script(
        script_content,
        reference_content,
        submission_content,
        timeout_seconds,
        arch=GPU_TO_SM["T4"],
    )


@app.function(
    gpu="T4",
    image=cuda_image,
)
def run_pytorch_script_t4(
    script_content: str,
    reference_content: str = None,
    submission_content: str = None,
    timeout_seconds: int = 600,
) -> tuple[str, float]:
    return run_pytorch_script(
        script_content,
        reference_content,
        submission_content,
        timeout_seconds,
        arch=GPU_TO_SM["T4"],
    )


# L4: sm_80 (L4 Tensor Core architecture)
@app.function(
    gpu="L4",
    image=cuda_image,
)
def run_cuda_script_l4(
    script_content: str,
    reference_content: str = None,
    submission_content: str = None,
    timeout_seconds: int = 600,
) -> tuple[str, float]:
    return run_cuda_script(
        script_content,
        reference_content,
        submission_content,
        timeout_seconds,
        arch=GPU_TO_SM["L4"],
    )


@app.function(
    gpu="L4",
    image=cuda_image,
)
def run_pytorch_script_l4(
    script_content: str,
    reference_content: str = None,
    submission_content: str = None,
    timeout_seconds: int = 600,
) -> tuple[str, float]:
    return run_pytorch_script(
        script_content,
        reference_content,
        submission_content,
        timeout_seconds,
        arch=GPU_TO_SM["L4"],
    )


# A100: sm_80 (Ampere architecture)
@app.function(
    gpu="A100",
    image=cuda_image,
)
def run_cuda_script_a100(
    script_content: str,
    reference_content: str = None,
    submission_content: str = None,
    timeout_seconds: int = 600,
) -> tuple[str, float]:
    return run_cuda_script(
        script_content,
        reference_content,
        submission_content,
        timeout_seconds,
        arch=GPU_TO_SM["A100"],
    )


@app.function(
    gpu="A100",
    image=cuda_image,
)
def run_pytorch_script_a100(
    script_content: str,
    reference_content: str = None,
    submission_content: str = None,
    timeout_seconds: int = 600,
) -> tuple[str, float]:
    return run_pytorch_script(
        script_content,
        reference_content,
        submission_content,
        timeout_seconds,
        arch=GPU_TO_SM["A100"],
    )


# H100: sm_90 (Hopper architecture)
@app.function(
    gpu="H100",
    image=cuda_image,
)
def run_cuda_script_h100(
    script_content: str,
    reference_content: str = None,
    submission_content: str = None,
    timeout_seconds: int = 600,
) -> tuple[str, float]:
    return run_cuda_script(
        script_content,
        reference_content,
        submission_content,
        timeout_seconds,
        arch=GPU_TO_SM["H100"],
    )


@app.function(
    gpu="H100",
    image=cuda_image,
)
def run_pytorch_script_h100(
    script_content: str,
    reference_content: str = None,
    submission_content: str = None,
    timeout_seconds: int = 600,
) -> tuple[str, float]:
    return run_pytorch_script(
        script_content,
        reference_content,
        submission_content,
        timeout_seconds,
        arch=GPU_TO_SM["H100"],
    )


def _get_runner_module_functions(prefix: str):
    current_module = sys.modules[__name__]
    return {
        name.split("_")[-1]: getattr(current_module, name)
        for name in dir(current_module)
        if name.startswith(f"run_{prefix}_script_")
    }


pytorch_function_map = _get_runner_module_functions("pytorch")
cuda_function_map = _get_runner_module_functions("cuda")


@contextmanager
def modal_context():
    """
    Context manager that ensures Modal functions are hydrated while in use.
    Usage:
        with hydrated_modal_runners() as runners:
            function = runners.get_runner("py", "t4")
            stdout, score = function(*args, **kwargs)
    """
    current_module = sys.modules[__name__]

    # Dynamically get all runner functions
    pytorch_functions = {
        name.split("_")[-1]: getattr(current_module, name)
        for name in dir(current_module)
        if name.startswith("run_pytorch_script_")
    }

    cuda_functions = {
        name.split("_")[-1]: getattr(current_module, name)
        for name in dir(current_module)
        if name.startswith("run_cuda_script_")
    }

    class Runners:
        def __init__(self):
            self._pytorch_map = pytorch_functions
            self._cuda_map = cuda_functions

        def get_runner(self, runner_type: str, gpu_type: str):
            if runner_type == "py":
                function = self._pytorch_map.get(gpu_type.lower())
            elif runner_type == "cu":
                function = self._cuda_map.get(gpu_type.lower())
            else:
                raise ValueError(f"Invalid runner type: {runner_type}")
            return function

        def _get_cuda_runner(self, gpu_type: str):
            function = self._cuda_map.get(gpu_type.lower())
            if function:
                return function
            raise ValueError(f"Function for gpu_type {gpu_type} not found")

        def _get_pytorch_runner(self, gpu_type: str):
            function = self._pytorch_map.get(gpu_type.lower())
            if function:
                return function
            raise ValueError(f"Function for gpu_type {gpu_type} not found")

    runners = Runners()
    try:
        yield runners
    finally:
        # Clean up if needed
        pass
