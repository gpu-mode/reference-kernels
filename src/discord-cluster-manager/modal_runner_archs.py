# This file contains wrapper functions for running
# Modal apps on specific devices. We will fix this later.


from consts import GPU_TO_SM
from modal_runner import app, cuda_image, modal_run_cuda_script, modal_run_pytorch_script


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
    return modal_run_cuda_script(
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
    return modal_run_pytorch_script(
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
    return modal_run_cuda_script(
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
    return modal_run_pytorch_script(
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
    return modal_run_cuda_script(
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
    return modal_run_pytorch_script(
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
    return modal_run_cuda_script(
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
    return modal_run_pytorch_script(
        script_content,
        reference_content,
        submission_content,
        timeout_seconds,
        arch=GPU_TO_SM["H100"],
    )
