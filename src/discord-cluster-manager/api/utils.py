import asyncio
from datetime import datetime

import requests
from consts import SubmissionMode, get_gpu_by_name
from env import (
    CLI_DISCORD_CLIENT_ID,
    CLI_DISCORD_CLIENT_SECRET,
    CLI_GITHUB_CLIENT_ID,
    CLI_GITHUB_CLIENT_SECRET,
    CLI_TOKEN_URL,
)
from fastapi import HTTPException
from report import RunProgressReporterAPI
from submission import SubmissionRequest, prepare_submission


class MockProgressReporter:
    """Class that pretends to be a progress reporter,
    is used to avoid errors when running submission,
    because runners report progress via discord interactions
    """

    async def push(self, message: str):
        pass

    async def update(self, message: str):
        pass


async def _handle_discord_oauth(code: str, redirect_uri: str) -> tuple[str, str]:
    """Handles the Discord OAuth code exchange and user info retrieval."""
    client_id = CLI_DISCORD_CLIENT_ID
    client_secret = CLI_DISCORD_CLIENT_SECRET
    token_url = CLI_TOKEN_URL
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
    client_id = CLI_GITHUB_CLIENT_ID
    client_secret = CLI_GITHUB_CLIENT_SECRET

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
    submission: SubmissionRequest, user_info: dict, mode: SubmissionMode, bot
):
    try:
        req = prepare_submission(submission, bot.leaderboard_db)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    selected_gpus = [get_gpu_by_name(gpu) for gpu in req.gpus]
    if len(selected_gpus) > 1 or selected_gpus[0] is None:
        raise HTTPException(status_code=400, detail="Invalid GPU type")

    command = bot.get_cog("SubmitCog").submit_leaderboard

    user_name = user_info["user_name"]
    user_id = user_info["user_id"]

    with bot.leaderboard_db as db:
        sub_id = db.create_submission(
            leaderboard=req.leaderboard,
            file_name=submission.file_name,
            code=submission.code,
            user_id=user_id,
            time=datetime.now(),
            user_name=user_name,
        )

    gpu = selected_gpus[0]

    try:
        tasks = [
            command(
                sub_id,
                submission.code,
                submission.file_name,
                gpu,
                RunProgressReporterAPI(),
                req.task,
                mode,
                None,
            )
        ]

        if mode == SubmissionMode.LEADERBOARD:
            tasks += [
                command(
                    sub_id,
                    submission.code,
                    submission.file_name,
                    gpu,
                    RunProgressReporterAPI(),
                    req.task,
                    SubmissionMode.PRIVATE,
                    req.secret_seed,
                )
            ]

        results = await asyncio.gather(*tasks)
    finally:
        with bot.leaderboard_db as db:
            db.mark_submission_done(sub_id)

    return results
