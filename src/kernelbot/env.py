import os
import types

from dotenv import load_dotenv

from libkernelbot.utils import get_github_branch_name


def init_environment():
    load_dotenv()

    # Validate environment
    required_env_vars = ["DISCORD_TOKEN", "GITHUB_TOKEN", "GITHUB_REPO"]
    for var in required_env_vars:
        if not os.getenv(var):
            raise ValueError(f"{var} not found")


init_environment()

env = types.SimpleNamespace()

# Discord-specific constants
env.DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
env.DISCORD_DEBUG_TOKEN = os.getenv("DISCORD_DEBUG_TOKEN")
env.DISCORD_CLUSTER_STAGING_ID = os.getenv("DISCORD_CLUSTER_STAGING_ID")
env.DISCORD_DEBUG_CLUSTER_STAGING_ID = os.getenv("DISCORD_DEBUG_CLUSTER_STAGING_ID")

# Only required to run the CLI against this instance
# setting these is required only to run the CLI against local instance
env.CLI_DISCORD_CLIENT_ID = os.getenv("CLI_DISCORD_CLIENT_ID", "")
env.CLI_DISCORD_CLIENT_SECRET = os.getenv("CLI_DISCORD_CLIENT_SECRET", "")
env.CLI_TOKEN_URL = os.getenv("CLI_TOKEN_URL", "")
env.CLI_GITHUB_CLIENT_ID = os.getenv("CLI_GITHUB_CLIENT_ID", "")
env.CLI_GITHUB_CLIENT_SECRET = os.getenv("CLI_GITHUB_CLIENT_SECRET", "")


# GitHub-specific constants
env.GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
env.GITHUB_REPO = os.getenv("GITHUB_REPO")
env.GITHUB_WORKFLOW_BRANCH = os.getenv("GITHUB_WORKFLOW_BRANCH", get_github_branch_name())
env.PROBLEMS_REPO = os.getenv("PROBLEMS_REPO")

# Directory that will be used for local problem development.
env.PROBLEM_DEV_DIR = os.getenv("PROBLEM_DEV_DIR", "examples")

# PostgreSQL-specific constants
env.POSTGRES_HOST = os.getenv("POSTGRES_HOST")
env.POSTGRES_DATABASE = os.getenv("POSTGRES_DATABASE")
env.POSTGRES_USER = os.getenv("POSTGRES_USER")
env.POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
env.POSTGRES_PORT = os.getenv("POSTGRES_PORT")
env.DATABASE_URL = os.getenv("DATABASE_URL")
env.DISABLE_SSL = os.getenv("DISABLE_SSL")
