import datetime
import functools
import logging
import subprocess
from typing import TYPE_CHECKING, Any, List, NotRequired, Optional, TypedDict

import discord
from consts import Language, SubmissionMode

if TYPE_CHECKING:
    from task import LeaderboardTask


def setup_logging(name: Optional[str] = None):
    """Configure and setup logging for the application"""

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name or __name__)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger


def with_error_handling(f: callable):
    @functools.wraps(f)
    async def wrap(self, interaction: discord.Interaction, *args, **kwargs):
        try:
            await f(self, interaction, *args, **kwargs)
        except KernelBotError as e:
            await send_discord_message(
                interaction,
                str(e),
                ephemeral=True,
            )
        except Exception as e:
            logging.exception("Unhandled exception %s", e, exc_info=e)
            await send_discord_message(
                interaction,
                "An unexpected error occurred. Please report this to the developers.",
                ephemeral=True,
            )

    return wrap


class KernelBotError(Exception):
    """
    This class represents an Exception that has been sanitized,
    i.e., whose message can be safely displayed to the user without
    risk of leaking internal bot details.
    """

    def __init__(self, message):
        super().__init__(message)


def get_github_branch_name():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().split("/", 1)[1]
    except subprocess.CalledProcessError:
        return "main"


async def get_user_from_id(bot, id) -> str:
    with bot.leaderboard_db as db:
        return db.get_user_from_id(id) or id


async def send_discord_message(
    interaction: discord.Interaction, msg: str, *, ephemeral=False, **kwargs
) -> None:
    """
    To get around response messages in slash commands that are
    called externally, send a message using the followup.
    """
    if interaction.response.is_done():
        await interaction.followup.send(msg, ephemeral=ephemeral, **kwargs)
    else:
        await interaction.response.send_message(msg, ephemeral=ephemeral, **kwargs)


async def send_logs(thread: discord.Thread, logs: str) -> None:
    """Send logs to a Discord thread, splitting by lines and respecting Discord's character limit.

    Args:
        thread: The Discord thread to send logs to
        logs: The log string to send
    """
    # Split logs into lines
    log_lines = logs.splitlines()

    current_chunk = []
    current_length = 0

    for line in log_lines:
        # Add 1 for the newline character
        line_length = len(line) + 1

        # If adding this line would exceed Discord's limit, send current chunk
        if current_length + line_length > 1990:  # Leave room for code block markers
            if current_chunk:
                chunk_text = "\n".join(current_chunk)
                await thread.send(f"```\n{chunk_text}\n```")
                current_chunk = []
                current_length = 0

        current_chunk.append(line)
        current_length += line_length

    # Send any remaining lines
    if current_chunk:
        chunk_text = "\n".join(current_chunk)
        await thread.send(f"```\n{chunk_text}\n```")


class LRUCache:
    def __init__(self, max_size: int):
        """LRU Cache implementation, as functools.lru doesn't work in async code
        Note: Implementation uses list for convenience because cache is small, so
        runtime complexity does not matter here.
        Args:
            max_size (int): Maximum size of the cache
        """
        self._cache = {}
        self._max_size = max_size
        self._q = []

    def __getitem__(self, key: Any, default: Any = None) -> Any | None:
        if key not in self._cache:
            return default

        self._q.remove(key)
        self._q.append(key)
        return self._cache[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self._cache:
            self._q.remove(key)
            self._q.append(key)
            self._cache[key] = value
            return

        if len(self._cache) >= self._max_size:
            self._cache.pop(self._q.pop(0))

        self._cache[key] = value
        self._q.append(key)

    def __contains__(self, key: Any) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def invalidate(self):
        """Invalidate the cache, clearing all entries, should be called when updating the underlying
        data in db
        """
        self._cache.clear()
        self._q.clear()


class LeaderboardItem(TypedDict):
    id: int
    name: str
    creator_id: int
    deadline: datetime.datetime
    task: "LeaderboardTask"
    gpu_types: List[str]
    forum_id: int
    secret_seed: NotRequired[int]


class LeaderboardRankedEntry(TypedDict):
    submission_id: int
    rank: int
    submission_name: str
    submission_time: datetime.datetime
    submission_score: float
    leaderboard_name: str
    user_id: int
    user_name: str
    gpu_type: str


class RunItem(TypedDict):
    start_time: datetime.datetime
    end_time: datetime.datetime
    mode: str
    secret: bool
    runner: str
    score: Optional[float]
    passed: bool
    compilation: dict
    meta: dict
    result: dict
    system: dict


class SubmissionItem(TypedDict):
    submission_id: int
    leaderboard_id: int
    leaderboard_name: str
    file_name: str
    user_id: int
    submission_time: datetime.datetime
    done: bool
    code: str
    runs: List[RunItem]


def build_task_config(
    task: "LeaderboardTask" = None,
    submission_content: str = None,
    arch: str = None,
    mode: SubmissionMode = None,
) -> dict:
    if task is None:
        assert mode == SubmissionMode.SCRIPT
        # TODO detect language
        lang = "py"

        config = {
            "lang": lang,
            "arch": arch,
        }

        eval_name = {"py": "eval.py", "cu": "eval.cu"}[lang]

        if lang == "py":
            config["main"] = "eval.py"

        return {
            **config,
            "sources": {
                eval_name: submission_content,
            },
        }
    else:
        all_files = {}
        for n, c in task.files.items():
            if c == "@SUBMISSION@":
                all_files[n] = submission_content
            else:
                all_files[n] = c

        common = {
            "lang": task.lang.value,
            "arch": arch,
            "benchmarks": task.benchmarks,
            "tests": task.tests,
            "mode": mode.value,
            "test_timeout": task.test_timeout,
            "benchmark_timeout": task.benchmark_timeout,
            "ranked_timeout": task.ranked_timeout,
            "ranking_by": task.ranking_by.value,
            "seed": task.seed,
        }

        if task.lang == Language.Python:
            return {
                "main": task.config.main,
                "sources": all_files,
                **common,
            }
        else:
            sources = {}
            headers = {}
            for f in all_files:
                if f in task.config.sources:
                    sources[f] = all_files[f]
                else:
                    headers[f] = all_files[f]

            return {
                "sources": sources,
                "headers": headers,
                "include_dirs": task.config.include_dirs,
            }


def format_time(value: float | str, err: Optional[float | str] = None, scale=None):  # noqa: C901
    # really ugly, but works for now
    value = float(value)

    scale = 1  # nanoseconds
    unit = "ns"
    if value > 2_000_000:
        scale = 1000_000
        unit = "ms"
    elif value > 2000:
        scale = 1000
        unit = "µs"

    value /= scale
    if err is not None:
        err = float(err)
        err /= scale
    if value < 1:
        if err:
            return f"{value} ± {err} {unit}"
        else:
            return f"{value} {unit}"
    elif value < 10:
        if err:
            return f"{value:.2f} ± {err:.3f} {unit}"
        else:
            return f"{value:.2f} {unit}"
    elif value < 100:
        if err:
            return f"{value:.1f} ± {err:.2f} {unit}"
        else:
            return f"{value:.1f} {unit}"
    else:
        if err:
            return f"{value:.0f} ± {err:.1f} {unit}"
        else:
            return f"{value:.0f} {unit}"
