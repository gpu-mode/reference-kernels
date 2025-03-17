import asyncio
import time
from dataclasses import asdict

from cogs.submit_cog import SubmitCog
from consts import _GPU_LOOKUP, SubmissionMode, get_gpu_by_name
from discord import app_commands
from fastapi import FastAPI, HTTPException, UploadFile
from utils import LeaderboardItem, build_task_config

app = FastAPI()

bot_instance = None

_last_action = time.time()
_submit_limiter = asyncio.Semaphore(3)


async def simple_rate_limit():
    """
    A very primitive rate limiter. This function returns at most
    10 times per second. Even if someone spams the API with
    requests, we're not hammering the bot.

    Note that there is no forward progress guarantee here:
    If we continually get new requests at a rate > 10/second,
    it is theoretically possible that some threads never exit the
    loop. We can worry about this as we scale up, and in any case
    it is better than hanging the discord bot.
    """
    global _last_action
    while time.time() - _last_action < 0.1:
        await asyncio.sleep(0.1)
    _last_action = time.time()
    return


def init_api(_bot_instance):
    global bot_instance
    bot_instance = _bot_instance


class MockProgressReporter:
    """Class that pretends to be a progress reporter,
    is used to avoid errors when running submission,
    because runners report progress via discord interactions
    """

    async def push(self, message: str):
        pass

    async def update(self, message: str):
        pass


@app.post("/{leaderboard_name}/{gpu_type}/{submission_mode}")
async def run_submission(
    leaderboard_name: str, gpu_type: str, submission_mode: str, file: UploadFile
) -> dict:
    """An endpoint that runs a submission on a given leaderboard, runner, and GPU type.

    Args:
        leaderboard_name (str): The name of the leaderboard to run the submission on.
        gpu_type (str): The type of GPU to run the submission on.
        file (UploadFile): The file to run the submission on.

    Raises:
        HTTPException: If the bot is not initialized.

    Returns:
        dict: A dictionary containing the status of the submission and the result.
        See class `FullResult` for more details.
    """
    await simple_rate_limit()

    submission_mode: SubmissionMode = SubmissionMode(submission_mode.lower())
    if submission_mode in [SubmissionMode.PROFILE]:
        raise HTTPException(status_code=400, detail="Profile submissions are not supported yet")

    if submission_mode not in [
        SubmissionMode.TEST,
        SubmissionMode.BENCHMARK,
        SubmissionMode.SCRIPT,
    ]:
        raise HTTPException(status_code=400, detail="Invalid submission mode")

    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")

    gpu_name = gpu_type.lower()
    gpu = get_gpu_by_name(gpu_name)
    if gpu is None:
        raise HTTPException(status_code=400, detail="Invalid GPU type")

    runner_name = gpu.runner.lower()

    cog_name = {"github": "GitHubCog", "modal": "ModalCog"}[runner_name]

    with bot_instance.leaderboard_db as db:
        leaderboard_item: LeaderboardItem = db.get_leaderboard(leaderboard_name)
        task = leaderboard_item["task"]

    runner_cog: SubmitCog = bot_instance.get_cog(cog_name)
    config = build_task_config(
        task=task,
        submission_content=file.file.read().decode("utf-8"),
        arch=runner_cog._get_arch(app_commands.Choice(name=gpu_name, value=gpu_name)),
        mode=submission_mode,
    )
    # limit the amount of concurrent submission by the API
    async with _submit_limiter:
        result = await runner_cog._run_submission(config, gpu, MockProgressReporter())

        return {"status": "success", "result": asdict(result)}


@app.get("/leaderboards")
async def get_leaderboards() -> list[LeaderboardItem]:
    """An endpoint that returns all leaderboards.

    Returns:
        list[LeaderboardItem]: A list of serialized `LeaderboardItem` objects,
        which hold information about the leaderboard, its deadline, its reference code,
        and the GPU types that are available for submissions.
    """
    await simple_rate_limit()
    with bot_instance.leaderboard_db as db:
        return db.get_leaderboards()


@app.get("/gpus/{leaderboard_name}")
async def get_gpus(leaderboard_name: str) -> list[str]:
    """An endpoint that returns all GPU types that are available for a given leaderboard and runner.

    Args:
        leaderboard_name (str): The name of the leaderboard to get the GPU types for.
        runner_name (str): The name of the runner to get the GPU types for.

    Returns:
        list[str]: A list of GPU types that are available for the given leaderboard and runner.
    """
    await simple_rate_limit()
    with bot_instance.leaderboard_db as db:
        gpu_types = db.get_leaderboard_gpu_types(leaderboard_name)

    runner_gpu_names = [gpu.name.lower() for gpu in _GPU_LOOKUP.values()]

    return [x for x in gpu_types if x.lower() in runner_gpu_names]


@app.get("/submissions/{leaderboard_name}/{gpu_name}")
async def get_submissions(
    leaderboard_name: str, gpu_name: str, limit: int = None, offset: int = 0
) -> list[dict]:
    await simple_rate_limit()
    with bot_instance.leaderboard_db as db:
        return db.get_leaderboard_submissions(
            leaderboard_name, gpu_name, limit=limit, offset=offset
        )


@app.get("/submission_count/{leaderboard_name}/{gpu_name}")
async def get_submission_count(leaderboard_name: str, gpu_name: str, user_id: str = None) -> dict:
    """Get the total count of submissions for pagination"""
    await simple_rate_limit()
    with bot_instance.leaderboard_db as db:
        count = db.get_leaderboard_submission_count(leaderboard_name, gpu_name, user_id)
        return {"count": count}
