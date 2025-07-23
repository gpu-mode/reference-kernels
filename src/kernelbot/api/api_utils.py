import requests
from fastapi import HTTPException

from kernelbot.env import env
from libkernelbot.backend import KernelBackend
from libkernelbot.consts import SubmissionMode
from libkernelbot.report import (
    Log,
    MultiProgressReporter,
    RunProgressReporter,
    RunResultReport,
    Text,
)
from libkernelbot.submission import SubmissionRequest, prepare_submission


async def _handle_discord_oauth(code: str, redirect_uri: str) -> tuple[str, str]:
    """Handles the Discord OAuth code exchange and user info retrieval."""
    client_id = env.CLI_DISCORD_CLIENT_ID
    client_secret = env.CLI_DISCORD_CLIENT_SECRET
    token_url = env.CLI_TOKEN_URL
    user_api_url = "https://discord.com/api/users/@me"

    if not client_id:
        raise HTTPException(status_code=500, detail="Discord client ID not configured.")
    if not client_secret:
        raise HTTPException(status_code=500, detail="Discord client secret not configured.")
    if not token_url:
        raise HTTPException(status_code=500, detail="Discord token URL not configured.")

    token_data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
    }

    try:
        token_response = requests.post(token_url, data=token_data)
        token_response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=401,
            detail=f"Failed to communicate with Discord token endpoint: {e}",
        ) from e

    token_json = token_response.json()
    access_token = token_json.get("access_token")
    if not access_token:
        raise HTTPException(
            status_code=401,
            detail=f"Failed to get access token from Discord: {token_response.text}",
        )

    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        user_response = requests.get(user_api_url, headers=headers)
        user_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=401,
            detail=f"Failed to communicate with Discord user endpoint: {e}",
        ) from e

    user_json = user_response.json()
    user_id = user_json.get("id")
    user_name = user_json.get("username")

    if not user_id or not user_name:
        raise HTTPException(
            status_code=500, detail="Failed to retrieve user ID or username from Discord."
        )

    return user_id, user_name


async def _handle_github_oauth(code: str, redirect_uri: str) -> tuple[str, str]:
    """Handles the GitHub OAuth code exchange and user info retrieval."""
    client_id = env.CLI_GITHUB_CLIENT_ID
    client_secret = env.CLI_GITHUB_CLIENT_SECRET

    token_url = "https://github.com/login/oauth/access_token"
    user_api_url = "https://api.github.com/user"

    if not client_id:
        raise HTTPException(status_code=500, detail="GitHub client ID not configured.")
    if not client_secret:
        raise HTTPException(status_code=500, detail="GitHub client secret not configured.")

    token_data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": redirect_uri,
    }
    print(token_data)
    headers = {"Accept": "application/json"}  # Request JSON response for token

    try:
        token_response = requests.post(token_url, data=token_data, headers=headers)
        token_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=401,
            detail=f"Failed to communicate with GitHub token endpoint: {e}",
        ) from e

    token_json = token_response.json()
    access_token = token_json.get("access_token")
    if not access_token:
        raise HTTPException(
            status_code=401,
            detail=f"Failed to get access token from GitHub: {token_response.text}",
        )

    auth_headers = {"Authorization": f"Bearer {access_token}"}
    try:
        user_response = requests.get(user_api_url, headers=auth_headers)
        user_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=401,
            detail=f"Failed to communicate with GitHub user endpoint: {e}",
        ) from e

    user_json = user_response.json()
    user_id = str(user_json.get("id"))  # GitHub ID is integer, convert to string for consistency
    user_name = user_json.get("login")  # GitHub uses 'login' for username

    if not user_id or not user_name:
        raise HTTPException(
            status_code=500, detail="Failed to retrieve user ID or username from GitHub."
        )

    return user_id, user_name


async def _run_submission(
    submission: SubmissionRequest, mode: SubmissionMode, backend: KernelBackend
):
    try:
        req = prepare_submission(submission, backend)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if len(req.gpus) != 1:
        raise HTTPException(status_code=400, detail="Invalid GPU type")

    reporter = MultiProgressReporterAPI()
    sub_id, results = await backend.submit_full(req, mode, reporter)
    return results, [rep.get_message() + "\n" + rep.long_report for rep in reporter.runs]


class MultiProgressReporterAPI(MultiProgressReporter):
    def __init__(self):
        self.runs = []

    async def show(self, title: str):
        return

    def add_run(self, title: str) -> "RunProgressReporterAPI":
        rep = RunProgressReporterAPI(title)
        self.runs.append(rep)
        return rep

    def make_message(self):
        return


class RunProgressReporterAPI(RunProgressReporter):
    def __init__(self, title: str):
        super().__init__(title=title)
        self.long_report = ""

    async def _update_message(self):
        pass

    async def display_report(self, title: str, report: RunResultReport):
        for part in report.data:
            if isinstance(part, Text):
                self.long_report += part.text
            elif isinstance(part, Log):
                self.long_report += f"\n\n## {part.header}:\n"
                self.long_report += f"```\n{part.content}```"
