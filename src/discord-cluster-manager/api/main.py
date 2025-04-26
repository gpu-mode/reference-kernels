import asyncio
import base64
import datetime
import json
import os
import time
from dataclasses import asdict
from typing import Annotated, Optional

from consts import SubmissionMode
from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from submission import SubmissionRequest
from utils import LeaderboardRankedEntry

from .utils import _handle_discord_oauth, _handle_github_oauth, _run_submission

app = FastAPI()


def json_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


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


async def validate_cli_header(
    x_popcorn_cli_id: Optional[str] = Header(None, alias="X-Popcorn-Cli-Id"),
) -> str:
    """
    FastAPI dependency to validate the X-Popcorn-Cli-Id header.

    Raises:
        HTTPException: If the header is missing or invalid.

    Returns:
        str: The validated user ID associated with the CLI ID.
    """
    if not x_popcorn_cli_id:
        raise HTTPException(status_code=400, detail="Missing X-Popcorn-Cli-Id header")

    if not bot_instance or not hasattr(bot_instance, "leaderboard_db"):
        raise HTTPException(
            status_code=500, detail="Bot instance or database not initialized for validation"
        )

    try:
        with bot_instance.leaderboard_db as db:
            if db is None:
                raise HTTPException(status_code=500, detail="Database connection failed")
            user_info = db.validate_cli_id(x_popcorn_cli_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error during validation: {e}") from e

    if user_info is None:
        raise HTTPException(status_code=401, detail="Invalid or unauthorized X-Popcorn-Cli-Id")

    return user_info


@app.get("/auth/init")
async def auth_init(provider: str) -> dict:
    if provider not in ["discord", "github"]:
        raise HTTPException(
            status_code=400, detail="Invalid provider, must be 'discord' or 'github'"
        )

    """
    Initialize authentication flow for the specified provider.
    Returns a random UUID to be used as state parameter in the OAuth flow.

    Args:
        provider (str): The authentication provider ('discord' or 'github')

    Returns:
        dict: A dictionary containing the state UUID
    """
    import uuid

    state_uuid = str(uuid.uuid4())

    # Ensure bot_instance and leaderboard_db are available
    if not bot_instance or not hasattr(bot_instance, "leaderboard_db"):
        raise HTTPException(status_code=500, detail="Bot instance or database not initialized")

    try:
        with bot_instance.leaderboard_db as db:
            if db is None:
                raise HTTPException(status_code=500, detail="Database connection failed")
            # Assuming init_user_from_cli exists and handles DB interaction
            db.init_user_from_cli(state_uuid, provider)
    except AttributeError as e:
        # Catch if leaderboard_db methods don't exist
        raise HTTPException(status_code=500, detail=f"Database interface error: {e}") from e
    except Exception as e:
        # Catch other potential errors during DB interaction
        raise HTTPException(status_code=500, detail=f"Failed to initialize auth in DB: {e}") from e

    return {"state": state_uuid}


@app.get("/auth/cli/{auth_provider}")
async def cli_auth(auth_provider: str, code: str, state: str):  # noqa: C901
    """
    Handle Discord/GitHub OAuth redirect. This endpoint receives the authorization code
    and state parameter from the OAuth flow.

    Args:
        auth_provider (str): 'discord' or 'github'
        code (str): Authorization code from OAuth provider
        state (str): Base64 encoded state containing cli_id and is_reset flag
    """

    if auth_provider not in ["discord", "github"]:
        raise HTTPException(
            status_code=400, detail="Invalid provider, must be 'discord' or 'github'"
        )

    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing authorization code or state")

    try:
        # Pad state if necessary for correct base64 decoding
        state_padded = state + "=" * (4 - len(state) % 4) if len(state) % 4 else state
        state_json = base64.urlsafe_b64decode(state_padded).decode("utf-8")
        state_data = json.loads(state_json)
        cli_id = state_data["cli_id"]
        is_reset = state_data.get("is_reset", False)
    except (json.JSONDecodeError, KeyError, Exception) as e:
        raise HTTPException(status_code=400, detail=f"Invalid state parameter: {e}") from None

    # Determine API URL (handle potential None value)
    api_base_url = os.environ.get("HEROKU_APP_DEFAULT_DOMAIN_NAME") or os.getenv("POPCORN_API_URL")
    if not api_base_url:
        raise HTTPException(
            status_code=500,
            detail="Redirect URI base not configured."
            "Set HEROKU_APP_DEFAULT_DOMAIN_NAME or POPCORN_API_URL.",
        )
    redirect_uri_base = api_base_url.rstrip("/")
    redirect_uri = f"https://{redirect_uri_base}/auth/cli/{auth_provider}"

    user_id = None
    user_name = None

    try:
        if auth_provider == "discord":
            user_id, user_name = await _handle_discord_oauth(code, redirect_uri)
        elif auth_provider == "github":
            user_id, user_name = await _handle_github_oauth(code, redirect_uri)

    except HTTPException as e:
        # Re-raise HTTPExceptions from helpers
        raise e
    except Exception as e:
        # Catch unexpected errors during OAuth handling
        raise HTTPException(
            status_code=500, detail=f"Error during {auth_provider} OAuth flow: {e}"
        ) from e

    if not user_id or not user_name:
        raise HTTPException(
            status_code=500, detail="Failed to retrieve user ID or username from provider."
        )

    if not bot_instance or not hasattr(bot_instance, "leaderboard_db"):
        raise HTTPException(
            status_code=500, detail="Bot instance or database not initialized for update"
        )

    try:
        with bot_instance.leaderboard_db as db:
            if is_reset:
                db.reset_user_from_cli(user_id, cli_id, auth_provider)
            else:
                db.create_user_from_cli(user_id, user_name, cli_id, auth_provider)

    except AttributeError as e:
        raise HTTPException(
            status_code=500, detail=f"Database interface error during update: {e}"
        ) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Database update failed: {e}") from e

    return {
        "status": "success",
        "message": f"Successfully authenticated via {auth_provider} and linked CLI ID.",
        "user_id": user_id,
        "user_name": user_name,
        "is_reset": is_reset,
    }


async def _stream_submission_response(
    submission_request, user_info, submission_mode_enum, bot_instance
):
    start_time = time.time()
    task: asyncio.Task | None = None
    try:
        task = asyncio.create_task(
            _run_submission(
                submission_request,
                user_info,
                submission_mode_enum,
                bot_instance,
            )
        )

        while not task.done():
            elapsed_time = time.time() - start_time
            yield f"event: status\ndata: {json.dumps({'status': 'processing',
                                                      'elapsed_time': round(elapsed_time, 2)}
                                                      ,default=json_serializer)}\n\n"

            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=15.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                yield f"event: error\ndata: {json.dumps(
                    {'status': 'error', 'detail': 'Submission cancelled'},
                    default=json_serializer)}\n\n"
                return

        result, reports = await task
        result_data = {
            "status": "success",
            "results": [asdict(r) for r in result],
            "reports": reports,
        }
        yield f"event: result\ndata: {json.dumps(result_data, default=json_serializer)}\n\n"

    except HTTPException as http_exc:
        error_data = {
            "status": "error",
            "detail": http_exc.detail,
            "status_code": http_exc.status_code,
        }
        yield f"event: error\ndata: {json.dumps(error_data, default=json_serializer)}\n\n"
    except Exception as e:
        error_type = type(e).__name__
        error_data = {
            "status": "error",
            "detail": f"An unexpected error occurred: {error_type}",
            "raw_error": str(e),
        }
        yield f"event: error\ndata: {json.dumps(error_data, default=json_serializer)}\n\n"
    finally:
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


@app.post("/{leaderboard_name}/{gpu_type}/{submission_mode}")
async def run_submission(  # noqa: C901
    leaderboard_name: str,
    gpu_type: str,
    submission_mode: str,
    file: UploadFile,
    user_info: Annotated[dict, Depends(validate_cli_header)],
) -> StreamingResponse:
    """An endpoint that runs a submission on a given leaderboard, runner, and GPU type.
    Streams status updates and the final result via Server-Sent Events (SSE).

    Requires a valid X-Popcorn-Cli-Id header.

    Args:
        leaderboard_name (str): The name of the leaderboard to run the submission on.
        gpu_type (str): The type of GPU to run the submission on.
        submission_mode (str): The mode for the submission (test, benchmark, etc.).
        file (UploadFile): The file to run the submission on.
        user_id (str): The validated user ID obtained from the X-Popcorn-Cli-Id header.

    Raises:
        HTTPException: If the bot is not initialized, or header/input is invalid.

    Returns:
        StreamingResponse: A streaming response containing the status and results of the submission.
    """
    await simple_rate_limit()
    user_name = user_info["user_name"]
    user_id = user_info["user_id"]

    try:
        submission_mode_enum: SubmissionMode = SubmissionMode(submission_mode.lower())
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Invalid submission mode value: '{submission_mode}'"
        ) from None

    if submission_mode_enum in [SubmissionMode.PROFILE]:
        raise HTTPException(
            status_code=400, detail="Profile submissions are not currently supported via API"
        )

    allowed_modes = [
        SubmissionMode.TEST,
        SubmissionMode.BENCHMARK,
        SubmissionMode.SCRIPT,
        SubmissionMode.LEADERBOARD,
    ]
    if submission_mode_enum not in allowed_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Submission mode '{submission_mode}' is not supported for this endpoint",
        )

    if not bot_instance:
        raise HTTPException(
            status_code=503, detail="Service temporarily unavailable: Bot not initialized"
        )

    try:
        with bot_instance.leaderboard_db as db:
            if db is None:
                raise HTTPException(
                    status_code=503,
                    detail="Service temporarily unavailable: Database connection failed",
                )
            leaderboard_item = db.get_leaderboard(leaderboard_name)
            if not leaderboard_item:
                all_leaderboards = [lb["name"] for lb in db.get_leaderboards()]
                if leaderboard_name not in all_leaderboards:
                    raise HTTPException(
                        status_code=404, detail=f"Leaderboard '{leaderboard_name}' not found."
                    )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error retrieving details for leaderboard '{leaderboard_name}'.",
                    )

            gpus = leaderboard_item.get("gpu_types", [])
            if gpu_type not in gpus:
                supported_gpus = ", ".join(gpus) if gpus else "None"
                raise HTTPException(
                    status_code=400,
                    detail=f"GPU type '{gpu_type}' is not supported for "
                    f"leaderboard '{leaderboard_name}'. Supported GPUs: {supported_gpus}",
                )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal server error while validating leaderboard/GPU: {e}"
        ) from e

    try:
        submission_content = await file.read()
        if not submission_content:
            raise HTTPException(
                status_code=400, detail="Empty file submitted. Please provide a file with code."
            )
        if len(submission_content) > 1_000_000:
            raise HTTPException(
                status_code=413, detail="Submission file is too large (limit: 1MB)."
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading submission file: {e}") from e

    try:
        submission_code = submission_content.decode("utf-8")
        submission_request = SubmissionRequest(
            code=submission_code,
            file_name=file.filename or "submission.py",
            user_id=user_id,
            gpus=[gpu_type],
            leaderboard=leaderboard_name,
        )
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400, detail="Failed to decode submission file content as UTF-8."
        ) from None
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal server error creating submission request: {e}"
        ) from e

    generator = _stream_submission_response(
        submission_request=submission_request,
        user_info={"user_id": user_id, "user_name": user_name},
        submission_mode_enum=submission_mode_enum,
        bot_instance=bot_instance,
    )

    return StreamingResponse(generator, media_type="text/event-stream")


@app.get("/leaderboards")
async def get_leaderboards():
    """An endpoint that returns all leaderboards.

    Returns:
        list[LeaderboardItem]: A list of serialized `LeaderboardItem` objects,
        which hold information about the leaderboard, its deadline, its reference code,
        and the GPU types that are available for submissions.
    """
    await simple_rate_limit()
    if not bot_instance or not hasattr(bot_instance, "leaderboard_db"):
        raise HTTPException(status_code=500, detail="Bot instance or database not initialized")
    try:
        with bot_instance.leaderboard_db as db:
            if db is None:
                raise HTTPException(status_code=500, detail="Database connection failed")
            return db.get_leaderboards()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching leaderboards: {e}") from e


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
    if not bot_instance or not hasattr(bot_instance, "leaderboard_db"):
        raise HTTPException(status_code=500, detail="Bot instance or database not initialized")

    try:
        with bot_instance.leaderboard_db as db:
            if db is None:
                raise HTTPException(status_code=500, detail="Database connection failed")

            # Validate leaderboard exists first
            leaderboard_names = [x["name"] for x in db.get_leaderboards()]
            if leaderboard_name not in leaderboard_names:
                raise HTTPException(status_code=400, detail="Invalid leaderboard name")

            gpu_types = db.get_leaderboard_gpu_types(leaderboard_name)
            if gpu_types is None:  # Handle case where function returns None
                return []

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching GPU data: {e}") from e

    # Filter based on known runners
    return gpu_types


@app.get("/submissions/{leaderboard_name}/{gpu_name}")
async def get_submissions(
    leaderboard_name: str, gpu_name: str, limit: int = None, offset: int = 0
) -> list[LeaderboardRankedEntry]:
    await simple_rate_limit()
    if not bot_instance or not hasattr(bot_instance, "leaderboard_db"):
        raise HTTPException(status_code=500, detail="Bot instance or database not initialized")
    try:
        with bot_instance.leaderboard_db as db:
            if db is None:
                raise HTTPException(status_code=500, detail="Database connection failed")
            # Add validation for leaderboard and GPU? Might be redundant if DB handles it.
            return db.get_leaderboard_submissions(
                leaderboard_name, gpu_name, limit=limit, offset=offset
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching submissions: {e}") from e


@app.get("/submission_count/{leaderboard_name}/{gpu_name}")
async def get_submission_count(leaderboard_name: str, gpu_name: str, user_id: str = None) -> dict:
    """Get the total count of submissions for pagination"""
    await simple_rate_limit()
    if not bot_instance or not hasattr(bot_instance, "leaderboard_db"):
        raise HTTPException(status_code=500, detail="Bot instance or database not initialized")
    try:
        with bot_instance.leaderboard_db as db:
            if db is None:
                raise HTTPException(status_code=500, detail="Database connection failed")
            count = db.get_leaderboard_submission_count(leaderboard_name, gpu_name, user_id)
            return {"count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching submission count: {e}") from e
