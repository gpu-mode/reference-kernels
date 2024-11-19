# Run modal token new
# Then run this script as a simple test

# Script executed successfully
# Stopping app - local entrypoint completed.
# tensor([1., 2., 3., 4., 5.], device='cuda:0')
# tensor([1., 2., 3., 4., 5.], device='cuda:0')
# tensor([ 2.,  4.,  6.,  8., 10.], device='cuda:0')

import modal

# Modal app setup
modal_app = modal.App("discord-bot-runner")

@modal_app.function(
    gpu="T4",
    image=modal.Image.debian_slim(python_version="3.12")
        .pip_install(["torch"])
)
async def run_script_on_modal():
    """
    Runs a Python script on Modal with GPU
    """
    try:
        # Define main script content
        main_script = """
import sys
import torch

# Your actual script content
a = torch.Tensor([1, 2, 3, 4, 5]).cuda()
b = torch.Tensor([1, 2, 3, 4, 5]).cuda()
print(a)
print(b)
print(a + b)
"""
        # Execute the script content directly
        exec(main_script, {'__name__': '__main__'})
        return "Script executed successfully"
    except Exception as e:
        return f"Error executing script: {str(e)}"

# Run the function
if __name__ == "__main__":
    with modal_app.run():
        result = run_script_on_modal.remote()
        print(result)