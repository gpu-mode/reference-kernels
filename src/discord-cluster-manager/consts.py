from enum import Enum, IntEnum
from typing import Type


class GPUType(Enum):
    NVIDIA = "nvidia_workflow.yml"
    AMD = "amd_workflow.yml"


class SchedulerType(Enum):
    GITHUB = "github"
    MODAL = "modal"
    SLURM = "slurm"


class GitHubGPU(Enum):
    NVIDIA = "nvidia"
    AMD = "amd"


class ModalGPU(Enum):
    T4 = "T4"
    L4 = "L4"
    A100 = "A100"
    H100 = "H100"


GPU_SELECTION = {"Modal": ModalGPU, "GitHub": GitHubGPU}


class ExitCode(IntEnum):
    """
    Exit codes for our runners. These are just the codes actively return,
    others are possible (e.g., exiting due to segfault, permissions, signal, ...)
    """

    # program ran successfully
    SUCCESS = 0
    # a cuda API call failed
    CUDA_FAIL = 110
    # could not setup file descriptor for custom pipe
    PIPE_FAILED = 111
    # didn't crash, but tests failed
    VALIDATE_FAIL = 112
    # problem parsing test/benchmark
    TEST_SPEC = 113
    # process was shut down because it timed out
    TIMEOUT_EXPIRED = 114


class SubmissionMode(Enum):
    """
    Different types of submission that can be made:
    Test: Run tests and give detailed results about passed/failed tests. These have short timeouts.
    Benchmark: Run larger benchmarks. Each benchmark is tested once, and then run multiple times.
    Profile: Gather profiling information. One selected benchmark is run under the profiler. No
        testing is performed in this mode (sometimes, you need to profile deliberately broken code)
    Leaderboard: Official submission to the leaderboard. This first runs public tests, then a
        repeated invocation of a single benchmark. Feedback for the secret benchmark is only very
        limited (no stdout/stderr).
    Private: Special run that does test followed by leaderboard (on a secret seed), but gives only
        very limited feedback.
    Script: Submit an arbitrary script.
    """

    TEST = "test"
    BENCHMARK = "benchmark"
    PROFILE = "profile"
    LEADERBOARD = "leaderboard"
    PRIVATE = "private"
    SCRIPT = "script"


class Language(Enum):
    Python = "py"
    CUDA = "cu"


def combine_enums(enums: list[Type[Enum]], combined_name: str) -> Enum:
    combined_members = {}
    for enum in enums:
        for name, member in enum.__members__.items():
            if name in combined_members:
                raise ValueError(f"Duplicate member name '{name}' found across Enums.")
            combined_members[name] = member.value

    return Enum(combined_name, combined_members)


AllGPU = combine_enums([ModalGPU, GitHubGPU], "AllGPU")


GPU_TO_SM = {
    "T4": "75",
    "L4": "80",
    "A100": "80",
    "H100": "90a",
}


# Modal-specific constants
MODAL_PATH = "/tmp/dcs/"
MODAL_EVAL_CODE_PATH = "/tmp/dcs/eval.py"
MODAL_REFERENCE_CODE_PATH = "/tmp/dcs/reference.py"
MODAL_SUBMISSION_CODE_PATH = "/tmp/dcs/submission.py"


# Compilation flags for Modal
CUDA_FLAGS = [
    "--std=c++20",
    "-DNDEBUG",
    "-Xcompiler=-Wno-psabi",
    "-Xcompiler=-fno-strict-aliasing",
    "--expt-extended-lambda",
    "--expt-relaxed-constexpr",
    "-forward-unknown-to-host-compiler",
    "-O3",
    "-Xnvlink=--verbose",
    "-Xptxas=--verbose",
    "-Xptxas=--warn-on-spills",
]
MODAL_CUDA_INCLUDE_DIRS = ["/ThunderKittens/include"]

NVIDIA_REQUIREMENTS = """
numpy
torch
setuptools
ninja
triton
"""

AMD_REQUIREMENTS = """
--index-url https://download.pytorch.org/whl/nightly/rocm6.2
pytorch-triton-rocm==3.1.0+cf34004b8a
torch==2.6.0.dev20241023+rocm6.2
"""
