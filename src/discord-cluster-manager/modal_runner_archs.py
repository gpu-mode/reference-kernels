# This file contains wrapper functions for running
# Modal apps on specific devices. We will fix this later.


from modal_runner import app, cuda_image, modal_run_config, python_image

gpus = ["T4", "L4", "A100", "H100"]
for gpu in gpus:
    app.function(gpu=gpu, image=cuda_image, name=f"run_cuda_script_{gpu.lower()}", serialized=True)(
        modal_run_config
    )
    app.function(
        gpu=gpu, image=python_image, name=f"run_pytorch_script_{gpu.lower()}", serialized=True
    )(modal_run_config)
