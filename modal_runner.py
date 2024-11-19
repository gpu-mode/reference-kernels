from modal import App, Image 

# Create a stub for the Modal app
modal_app = App("discord-bot-runner")

@modal_app.function(
    gpu="T4",
    image=Image.debian_slim(python_version="3.10").pip_install(["torch"])
)
def run_script(script_content: str) -> str:
    """
    Executes the provided Python script in an isolated environment
    """
    import sys
    from io import StringIO
    
    # Capture stdout
    output = StringIO()
    sys.stdout = output
    
    try:
        # Create a new dictionary for local variables to avoid polluting the global namespace
        local_vars = {}
        # Execute the script in the isolated namespace
        exec(script_content, {}, local_vars)
        return output.getvalue()
    except Exception as e:
        return f"Error executing script: {str(e)}"
    finally:
        sys.stdout = sys.__stdout__

# For testing the Modal function directly
if __name__ == "__main__":
    with modal_app.run():
        result = run_script.remote("print('Hello from Modal!')")
        print(result)