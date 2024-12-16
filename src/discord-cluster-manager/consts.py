import os
from enum import Enum

from dotenv import load_dotenv


def init_environment():
    load_dotenv()

    # Validate environment
    required_env_vars = ["DISCORD_TOKEN", "GITHUB_TOKEN", "GITHUB_REPO"]
    for var in required_env_vars:
        if not os.getenv(var):
            raise ValueError(f"{var} not found")


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


init_environment()
# Discord-specific constants
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_DEBUG_TOKEN = os.getenv("DISCORD_DEBUG_TOKEN")
DISCORD_CLUSTER_STAGING_ID = os.getenv("DISCORD_CLUSTER_STAGING_ID")
DISCORD_DEBUG_CLUSTER_STAGING_ID = os.getenv("DISCORD_DEBUG_CLUSTER_STAGING_ID")

# GitHub-specific constants
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")

# PostgreSQL-specific constants
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_DATABASE = os.getenv("POSTGRES_DATABASE")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
DATABASE_URL = os.getenv("DATABASE_URL")
