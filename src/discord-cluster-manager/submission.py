import copy
import dataclasses
from datetime import datetime
from typing import Optional, Union

from better_profanity import profanity
from leaderboard_db import LeaderboardDB
from task import LeaderboardTask
from utils import KernelBotError, LeaderboardItem


@dataclasses.dataclass
class SubmissionRequest:
    # to be filled in when making the request
    code: str
    file_name: str
    user_id: int
    gpus: Union[None, str, list]
    leaderboard: Optional[str]


@dataclasses.dataclass
class ProcessedSubmissionRequest(SubmissionRequest):
    task: LeaderboardTask
    secret_seed: int
    task_gpus: list


def prepare_submission(req: SubmissionRequest, lb_db: LeaderboardDB) -> ProcessedSubmissionRequest:
    if profanity.contains_profanity(req.file_name):
        raise KernelBotError("Please provide a non rude filename")

    # check file extension
    if not req.file_name.endswith((".py", ".cu", ".cuh", ".cpp")):
        raise KernelBotError(
            "Please provide a Python (.py) or CUDA (.cu / .cuh / .cpp) file",
        )

    # process file directives
    req = handle_popcorn_directives(req)
    assert req.leaderboard is not None

    leaderboard = lookup_leaderboard(req.leaderboard, lb_db)
    check_deadline(leaderboard)

    task_gpus = get_avail_gpus(req.leaderboard, lb_db)

    if req.gpus is not None:
        for g in req.gpus:
            if g not in task_gpus:
                task_gpu_list = "".join([f" * {t}\n" for t in task_gpus])
                raise KernelBotError(
                    f"GPU {g} not available for `{req.leaderboard}`\n"
                    f"Choose one of: {task_gpu_list}",
                )
    elif len(task_gpus) == 1:
        req.gpus = task_gpus

    return ProcessedSubmissionRequest(
        **dataclasses.asdict(req),
        task=leaderboard["task"],
        secret_seed=leaderboard["secret_seed"],
        task_gpus=task_gpus,
    )


def lookup_leaderboard(leaderboard: str, lb_db: LeaderboardDB) -> LeaderboardItem:
    with lb_db as db:
        leaderboard_item = db.get_leaderboard(leaderboard)
        if not leaderboard_item:
            raise KernelBotError(f"Leaderboard {leaderboard} not found.")
        return leaderboard_item


def check_deadline(leaderboard: LeaderboardItem):
    now = datetime.now()
    deadline = leaderboard["deadline"]

    if now.date() > deadline.date():
        raise KernelBotError(
            f"The deadline to submit to {leaderboard['name']} has passed.\n"
            f"It was {deadline.date()} and today is {now.date()}."
        )


def get_avail_gpus(leaderboard: str, lb_db: LeaderboardDB):
    """
    Returns the list of available GPUs for a task.
    """
    with lb_db as db:
        gpus = db.get_leaderboard_gpu_types(leaderboard)

    if len(gpus) == 0:
        raise KernelBotError("❌ No available GPUs for Leaderboard " + f"`{leaderboard}`.")

    return gpus


def handle_popcorn_directives(req: SubmissionRequest) -> SubmissionRequest:
    req = copy.deepcopy(req)
    info = _get_popcorn_directives(req.code)
    # command argument GPUs overwrites popcorn directive
    if info["gpus"] is not None and req.gpus is None:
        req.gpus = info["gpus"]

    if info["leaderboard"] is not None:
        if req.leaderboard is not None:
            pass
            # TODO how do we handle this?
            # await send_discord_message(
            #     interaction,
            #     "Leaderboard name specified in the command doesn't match the one "
            #     f"in the submission script header. Submitting to `{leaderboard_name}`",
            #     ephemeral=True,
            # )
        else:
            req.leaderboard = info["leaderboard"]

    if req.leaderboard is None:
        raise KernelBotError(
            "Missing leaderboard name. "
            "Either supply one as an argument in the submit command, or "
            "specify it in your submission script using the "
            "`{#,//}!POPCORN leaderboard <leaderboard_name>` directive.",
        )

    return req


def _get_popcorn_directives(submission: str) -> dict:
    popcorn_info = {"gpus": None, "leaderboard": None}
    for line in submission.splitlines():
        # only process the first comment block of the file.
        # for simplicity, don't care whether these are python or C++ comments here
        if not (line.startswith("//") or line.startswith("#")):
            break

        args = line.split()
        if args[0] in ["//!POPCORN", "#!POPCORN"]:
            arg = args[1].strip().lower()
            if arg in ["gpu", "gpus"]:
                popcorn_info["gpus"] = args[2:]
            elif arg == "leaderboard":
                popcorn_info["leaderboard"] = args[2]
    return popcorn_info
