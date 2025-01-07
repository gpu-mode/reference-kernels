# This file contains wrapper functions for running
# Modal apps on specific devices. We will fix this later.

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


pytorch_function_map = {
    "t4": run_pytorch_script_t4,
    "l4": run_pytorch_script_l4,
    "a100": run_pytorch_script_a100,
    "h100": run_pytorch_script_h100,
}

cuda_function_map = {
    "t4": run_cuda_script_t4,
    "l4": run_cuda_script_l4,
    "a100": run_cuda_script_a100,
    "h100": run_cuda_script_h100,
}


def get_pytorch_modal_runner(gpu_type: str):
    function = pytorch_function_map.get(gpu_type.lower())

    if function:
        return function
    else:
        raise ValueError(f"Function for gpu_type {gpu_type} not found")


def get_cuda_modal_runner(gpu_type: str):
    function = cuda_function_map.get(gpu_type.lower())

    if function:
        return function
    else:
        raise ValueError(f"Function for gpu_type {gpu_type} not found")
