import os

from dotenv import load_dotenv


def init_environment():
    load_dotenv()

    # Validate environment
    required_env_vars = ["DISCORD_TOKEN", "GITHUB_TOKEN", "GITHUB_REPO"]
    for var in required_env_vars:
        if not os.getenv(var):
            raise ValueError(f"{var} not found")


init_environment()

# Discord-specific constants
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_DEBUG_TOKEN = os.getenv("DISCORD_DEBUG_TOKEN")
DISCORD_CLUSTER_STAGING_ID = os.getenv("DISCORD_CLUSTER_STAGING_ID")
DISCORD_DEBUG_CLUSTER_STAGING_ID = os.getenv("DISCORD_DEBUG_CLUSTER_STAGING_ID")

# GitHub-specific constants
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")
PROBLEMS_REPO = os.getenv("PROBLEMS_REPO")

# Directory that will be used for local problem development.
PROBLEM_DEV_DIR = os.getenv("PROBLEM_DEV_DIR", "examples")

# PostgreSQL-specific constants
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_DATABASE = os.getenv("POSTGRES_DATABASE")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
DATABASE_URL = os.getenv("DATABASE_URL")
DISABLE_SSL = os.getenv("DISABLE_SSL")
