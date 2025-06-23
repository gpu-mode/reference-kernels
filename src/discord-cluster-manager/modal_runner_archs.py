# This file contains wrapper functions for running
# Modal apps on specific devices. We will fix this later.
from modal_runner import app, cuda_image, modal_run_config

gpus = ["T4", "L4", "A100-80GB", "H100!", "B200"]
for gpu in gpus:
    gpu_slug = gpu.lower().split("-")[0].strip("!")
    app.function(gpu=gpu, image=cuda_image, name=f"run_cuda_script_{gpu_slug}", serialized=True)(
        modal_run_config
    )
    app.function(gpu=gpu, image=cuda_image, name=f"run_pytorch_script_{gpu_slug}", serialized=True)(
        modal_run_config
    )
