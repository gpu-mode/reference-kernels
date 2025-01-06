from enum import Enum


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


# Modal-specific constants
MODAL_PATH = "/tmp/dcs/"
MODAL_EVAL_CODE_PATH = "/tmp/dcs/eval.py"
MODAL_REFERENCE_CODE_PATH = "/tmp/dcs/reference.py"
MODAL_SUBMISSION_CODE_PATH = "/tmp/dcs/train.py"
