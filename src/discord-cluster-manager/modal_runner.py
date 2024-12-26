import signal
from contextlib import contextmanager

from modal import App, Image

# Create a stub for the Modal app
# IMPORTANT: This has to stay in separate file or modal breaks
modal_app = App("discord-bot-runner")


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


@modal_app.function(gpu="T4", image=Image.debian_slim(python_version="3.10").pip_install(["torch"]))
def run_pytorch_script(script_content: str, timeout_seconds: int = 300) -> tuple[str, float]:
    """
    Executes the provided PyTorch GPU kernel in an isolated environment with a timeout

    Args:
        script_content: The PyTorch script containing the GPU kernel to benchmark
        timeout_seconds: Maximum execution time before timeout (default: 300 seconds)

    Returns:
        tuple[str, float]: (Kernel output, execution time in milliseconds)

    NOTE: Modal execution time is not programmatically accessible, so we manually calculate it
    """
    import sys
    import time
    from io import StringIO

    # Capture stdout
    output = StringIO()
    sys.stdout = output

    try:
        with timeout(timeout_seconds):
            # Create a new dictionary for local variables to avoid polluting the global namespace
            local_vars = {}

            execution_start_time = time.perf_counter()

            # Execute the script in the isolated namespace
            exec(script_content, {}, local_vars)

            execution_end_time = time.perf_counter()

            execution_time_ms = (execution_end_time - execution_start_time) * 1000

        return output.getvalue(), execution_time_ms

    except TimeoutException as e:
        return f"Timeout Error: {str(e)}", 0.0
    except Exception as e:
        return f"Error executing script: {str(e)}", 0.0
    finally:
        sys.stdout = sys.__stdout__


@modal_app.function(
    gpu="T4",
    image=Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu24.04", add_python="3.11"),
)
def run_cuda_script(script_content: str, timeout_seconds: int = 600) -> tuple[str, float]:
    """
    Executes the provided CUDA kernel in an isolated environment with a timeout

    Args:
        script_content: The CUDA script containing the GPU kernel
        timeout_seconds: Maximum execution time in seconds (default: 600 seconds)

    Returns:
        tuple[str, float]: (Kernel output, execution time in milliseconds)

    NOTE: Modal execution time is not programmatically accessible, so we manually calculate it
    """
    import os
    import subprocess
    import sys
    import time
    from io import StringIO

    # Capture stdout
    output = StringIO()
    sys.stdout = output

    try:
        with timeout(timeout_seconds):
            execution_start_time = time.perf_counter()

            # Compile the CUDA code
            with open("script.cu", "w") as f:
                f.write(script_content)

            compile_process = subprocess.run(
                ["nvcc", "script.cu", "-o", "script.out"],
                capture_output=True,
                text=True,
            )

            if compile_process.returncode != 0:
                return f"Compilation Error:\n{compile_process.stderr}", 0.0

            run_process = subprocess.run(["./script.out"], capture_output=True, text=True)
            execution_end_time = time.perf_counter()

            execution_time_sec = execution_end_time - execution_start_time
            execution_time_ms = execution_time_sec * 1000

            return run_process.stdout, execution_time_ms

    except TimeoutException as e:
        return f"Timeout Error: {str(e)}", 0.0
    except Exception as e:
        return f"Error: {str(e)}", 0.0
    finally:
        if os.path.exists("script.cu"):
            os.remove("script.cu")
        if os.path.exists("script.out"):
            os.remove("script.out")
        sys.stdout = sys.__stdout__
