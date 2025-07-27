import copy
import dataclasses
import math
import typing
from datetime import datetime
from typing import Optional, Union

from better_profanity import profanity

from libkernelbot.consts import RankCriterion
from libkernelbot.leaderboard_db import LeaderboardDB, LeaderboardItem
from libkernelbot.run_eval import FullResult
from libkernelbot.task import LeaderboardTask
from libkernelbot.utils import KernelBotError, setup_logging

if typing.TYPE_CHECKING:
    from backend import KernelBackend


logger = setup_logging(__name__)


@dataclasses.dataclass
class SubmissionRequest:
    # to be filled in when making the request
    code: str
    file_name: str
    user_id: int
    user_name: str
    gpus: Union[None, str, list]
    leaderboard: Optional[str]


@dataclasses.dataclass
class ProcessedSubmissionRequest(SubmissionRequest):
    task: LeaderboardTask
    secret_seed: int
    task_gpus: list


def prepare_submission(
    req: SubmissionRequest, backend: "KernelBackend"
) -> ProcessedSubmissionRequest:
    if not backend.accepts_jobs:
        raise KernelBotError(
            "The bot is currently not accepting any new submissions, please try again later."
        )

    if profanity.contains_profanity(req.file_name):
        raise KernelBotError("Please provide a non-rude filename")

    # check file extension
    if not req.file_name.endswith((".py", ".cu", ".cuh", ".cpp")):
        raise KernelBotError(
            "Please provide a Python (.py) or CUDA (.cu / .cuh / .cpp) file",
        )

    # process file directives
    req = handle_popcorn_directives(req)
    assert req.leaderboard is not None

    with backend.db as db:
        leaderboard = db.get_leaderboard(req.leaderboard)
    check_deadline(leaderboard)

    task_gpus = get_avail_gpus(req.leaderboard, backend.db)

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
        raise KernelBotError(f"❌ No available GPUs for Leaderboard `{leaderboard}`.")

    return gpus


def handle_popcorn_directives(req: SubmissionRequest) -> SubmissionRequest:
    req = copy.deepcopy(req)
    info = _get_popcorn_directives(req.code)
    # command argument GPUs overwrites popcorn directive
    if info["gpus"] is not None and req.gpus is None:
        req.gpus = info["gpus"]

    if info["leaderboard"] is not None:
        if req.leaderboard is not None and req.leaderboard != info["leaderboard"]:
            raise KernelBotError(
                f"Leaderboard name `{req.leaderboard}` specified in the command"
                f" doesn't match the one "
                f"in the submission script header `{info['leaderboard']}`."
            )
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


def _get_popcorn_directives(submission: str) -> dict:  # noqa: C901
    popcorn_info = {"gpus": None, "leaderboard": None}
    for line in submission.splitlines():
        # only process the first comment block of the file.
        # for simplicity, don't care whether these are python or C++ comments here
        if not (line.startswith("//") or line.startswith("#")):
            break

        args = line.split()
        if args[0] in ["//!POPCORN", "#!POPCORN"]:
            arg = args[1].strip().lower()
            if len(args) < 3:
                raise KernelBotError(f"!POPCORN directive missing argument: {line}")
            #  allow both versions of the argument
            if arg == "gpu":
                arg = "gpus"

            if arg not in popcorn_info:
                raise KernelBotError(f"Invalid !POPCORN directive: {arg}")

            if popcorn_info[arg] is not None:
                raise KernelBotError(f"Found multiple values for !POPCORN directive {arg}")

            if arg == "gpus":
                popcorn_info["gpus"] = args[2:]
            elif arg == "leaderboard":
                popcorn_info["leaderboard"] = args[2].strip()
                if len(popcorn_info["leaderboard"]) == 0:
                    raise KernelBotError(
                        "No leaderboard specified in !POPCORN Leaderboard directive"
                    )
    return popcorn_info


def compute_score(result: FullResult, task: LeaderboardTask, submission_id: int) -> float:
    num_benchmarks = int(result.runs["leaderboard"].run.result["benchmark-count"])
    if task.ranking_by == RankCriterion.LAST:
        if num_benchmarks != 1:
            logger.error(
                "Ranked submission error for submission %d ranking_by is `last`, "
                "but got %d benchmarks",
                submission_id,
                num_benchmarks,
            )
            raise KernelBotError(
                f"Expected submission to have exactly one benchmark," f"got {num_benchmarks}."
            )
        score = float(result.runs["leaderboard"].run.result["benchmark.0.mean"]) / 1e9
    else:
        scores = []
        for i in range(num_benchmarks):
            scores.append(float(result.runs["leaderboard"].run.result[f"benchmark.{i}.mean"]) / 1e9)
        if task.ranking_by == RankCriterion.MEAN:
            score = sum(scores) / len(scores)
        elif task.ranking_by == RankCriterion.GEOM:
            score = math.pow(math.prod(scores), 1.0 / num_benchmarks)
        else:
            raise KernelBotError(f"Invalid submission mode {task.ranking_by}")

    return score
