from modal import App, Image
import signal
from contextlib import contextmanager

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



@modal_app.function(
    gpu="T4", image=Image.debian_slim(python_version="3.10").pip_install(["torch"])
)
def run_script(script_content: str, timeout_seconds: int = 300) -> str:
    """
    Executes the provided Python script in an isolated environment with a timeout

    Args:
        script_content: The Python script to execute
        timeout_seconds: Maximum execution time in seconds (default: 300 seconds / 5 minutes)

    Returns:
        str: Output of the script or error message
    """
    import sys
    from io import StringIO

    # Capture stdout
    output = StringIO()
    sys.stdout = output

    try:
        with timeout(timeout_seconds):
            # Create a new dictionary for local variables to avoid polluting the global namespace
            local_vars = {}
            # Execute the script in the isolated namespace
            exec(script_content, {}, local_vars)
        return output.getvalue()

    except TimeoutException as e:
        return f"Timeout Error: {str(e)}"
    except Exception as e:
        return f"Error executing script: {str(e)}"
    finally:
        sys.stdout = sys.__stdout__
