import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

from consts import CUDA_FLAGS


@dataclass
class CompileResult:
    # fmt: off
    nvcc_found: bool    # did we find nvcc?
    nvcc_version: str   # the result of nvcc --version
    success: bool       # did it compile successfully
    command: str        # the command that was run to compile the code
    stdout: str         # standard output produced by the compiler
    stderr: str         # standard error produced by the compiler
    exit_code: int      # exit code produced by the compiler
    # fmt: on


def _make_cmd(args: list[str]):
    return " ".join(map(shlex.quote, args))


def compile_cuda_script(  # # noqa: C901
    files: list[str],
    arch: int = None,
    include_dirs: list[str] = None,
    verbose: bool = False,
) -> CompileResult:
    """
    Compiles a set of cuda files with nvcc.

    Args:
        files: List of files to compile.
        arch: Architecture to compile for. If None, uses `native`
        include_dirs: additional include directories to supply to nvcc
        verbose: whether to print progress or be silent

    Returns:
        A `CompileResult` that summarizes the compilation process.

    """
    if include_dirs is None:
        include_dirs = []

    if verbose:
        print_ = print
    else:
        print_ = lambda *args, **kwargs: None

    # Check CUDA is available and installed correctly
    print_("[CUDA Env Check]")
    try:
        # these check cuda compiler is also available
        nvcc = subprocess.check_output(["which", "nvcc"], encoding="utf-8").strip()
        nvcc_version = subprocess.check_output(["nvcc", "--version"], encoding="utf-8")
    except subprocess.CalledProcessError as e:
        return CompileResult(
            nvcc_found=False,
            success=False,
            nvcc_version='',
            command=_make_cmd(e.cmd),
            stdout=e.stdout,
            stderr=e.stderr,
            exit_code=e.returncode,
        )

    if arch is None:
        ARCH = "-arch=native"
    else:
        ARCH = f"-gencode=arch=compute_{arch},code=sm_{arch}"

    command = [nvcc] + CUDA_FLAGS + include_dirs + files + [ARCH, "-o", "eval.out"]

    print_("[Compiling]")
    try:
        compile_process = subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        return CompileResult(
            nvcc_found=True,
            success=False,
            nvcc_version=nvcc_version,
            command=_make_cmd(e.cmd),
            stdout=e.stdout,
            stderr=e.stderr,
            exit_code=e.returncode,
        )
    return CompileResult(
        nvcc_found=True,
        success=True,
        nvcc_version=nvcc_version,
        command=_make_cmd(compile_process.args),
        stdout=compile_process.stdout,
        stderr=compile_process.stderr,
        exit_code=compile_process.returncode,
    )


def run_cuda_script(  # # noqa: C901
    script_content: str,
    reference_content: str = None,
    submission_content: str = None,
    arch: int = None,
    include_dirs: list[str] = None,
) -> tuple[str, float]:
    """
    Executes the provided CUDA kernel in an isolated environment with a timeout

    Args:
        script_content: The CUDA script containing the GPU kernel
        reference_content: The (optional) reference code, used for leaderboards.
        submission_content: The (optional) submission code, used for leaderboards.
        arch: The arch code for the compute/sm versions. If None, native arch is used.
        include_dirs: Additional include directories, e.g., for thunderkittens/cutlass etc

    Returns:
        tuple[str, float]: (Kernel output, execution time in milliseconds)
    """
    if include_dirs is None:
        include_dirs = []

    try:
        # Write submission files to directory
        if reference_content is not None:
            with open("reference.cuh", "w") as f:
                f.write(reference_content)

        if submission_content is not None:
            with open("train.cuh", "w") as f:
                f.write(submission_content)

        with open("eval.cu", "w") as f:
            f.write(script_content)

        compile_result = compile_cuda_script(
            files=["eval.cu"],
            arch=arch,
            include_dirs=include_dirs,
            verbose=True,
        )

        if not compile_result.success:
            raise RuntimeError(
                "CUDA compilation failed with return code "
                + f"{compile_result.exit_code}:\n{compile_result.stderr}\n"
                + f"compile command. `{compile_result.command}`"
            )

        # set up a pipe so the tester can communicate its verdict with us
        env = os.environ.copy()
        pipe_read, pipe_write = os.pipe()
        env["POPCORN_FD"] = str(pipe_write)

        execution_start_time = time.perf_counter()
        run_process = subprocess.run(
            ["./eval.out"],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            pass_fds=[pipe_write],
        )
        # terminate output writing
        os.close(pipe_write)
        # and fetch pipe's content
        result = os.fdopen(pipe_read, "r").read()
        execution_end_time = time.perf_counter()

        print("result", result)
        print("run process stdout", run_process.stdout)
        print("run process stderr", run_process.stderr)

        score = None
        passed = None
        for line in result.splitlines():
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if key == "duration.mean":
                score = float(value) / 1e9
            elif key == "duration.std":
                _ = float(value) / 1e9
            elif key == "duration.err":
                _ = float(value) / 1e9
            elif key == "duration.best":
                _ = float(value) / 1e9
            elif key == "duration.worst":
                _ = float(value) / 1e9
            elif key == "check":
                passed = value == "pass"
            else:
                print(f"unknown key {key} = {value}")
        # TODO: handle the case when "check" key is missing?
        if not passed:
            return "check_implementation failed", 0.0

        if score is None:
            score = execution_end_time - execution_start_time
            if passed:
                return "check_implementation failed", 0.0
            else:
                # This case isn't handled well in modal_cog
                return None, score

        return result, score

    except subprocess.CalledProcessError as e:
        return f"Error executing script: {str(e)}\n{e.stderr}", 0.0
    except Exception as e:
        return f"Error executing script: {str(e)}", 0.0
    finally:
        tmp_files = ["reference.cuh", "train.cuh", "eval.cu", "eval.out"]
        for f in tmp_files:
            if os.path.exists(f):
                os.remove(f)


def run_pytorch_script(  # noqa: C901
    script_content: str,
    reference_content: Optional[str] = None,
    submission_content: Optional[str] = None,
    arch: int = None,
) -> tuple[str, float]:
    """
    Executes the provided PyTorch GPU kernel in an isolated environment with a timeout

    Args:
        script_content: The PyTorch script containing the GPU kernel to benchmark
        reference_content: The (optional) reference code, used for leaderboards.
        submission_content: The (optional) submission code, used for leaderboards.
        arch: The arch code for the compute/sm versions.

    Returns:
        tuple[str, float]: (Kernel output, execution time in milliseconds)
    """
    try:
        # Write submission files to directory
        if reference_content is not None:
            with open("reference.py", "w") as f:
                f.write(reference_content)

        if submission_content is not None:
            with open("train.py", "w") as f:
                f.write(submission_content)

        with open("eval.py", "w") as f:
            f.write(script_content)

        execution_start_time = time.perf_counter()
        result = subprocess.run(
            ["python", "eval.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Script execution failed with return code "
                + f"{result.returncode}:\n{result.stderr}"
            )

        score = None
        for line in result.stdout.splitlines():
            if line.startswith("score:"):
                score = float(line.split(":")[1].strip())
                return "score", score

        if score is None:
            execution_end_time = time.perf_counter()
            score = execution_end_time - execution_start_time

        return result.stdout, score
    except Exception as e:
        return f"Error executing script: {str(e)}", 0.0
    finally:
        tmp_files = ["eval.py", "reference.py", "train.py"]
        for f in tmp_files:
            if os.path.exists(f):
                os.remove(f)
