from enum import Enum
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
    "T4": 75,
    "L4": 80,
    "A100": 80,
    "H100": 90,
}


# Modal-specific constants
MODAL_PATH = "/tmp/dcs/"
MODAL_EVAL_CODE_PATH = "/tmp/dcs/eval.py"
MODAL_REFERENCE_CODE_PATH = "/tmp/dcs/reference.py"
MODAL_SUBMISSION_CODE_PATH = "/tmp/dcs/train.py"


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
MODAL_CUDA_INCLUDE_DIRS = ["-I/ThunderKittens/include"]
