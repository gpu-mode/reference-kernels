import datetime
import logging
import subprocess
from typing import Any, List, TypedDict, Optional

import discord
from consts import Language, SubmissionMode
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


async def get_user_from_id(id, interaction, bot):
    # This currently doesn't work.
    if interaction.guild:
        # In a guild, try to get the member by ID
        try:
            member = await interaction.guild.fetch_member(id)
        except Exception:
            member = id

        return member
    else:
        # If the interaction is in DMs, we can get the user directly
        user = await bot.fetch_user(id)
        if user:
            username = user.global_name if member.nick is None else member.nick
            return username
        else:
            return id


async def send_discord_message(interaction: discord.Interaction, msg: str, **kwargs) -> None:
    """
    To get around response messages in slash commands that are
    called externally, send a message using the followup.
    """
    if interaction.response.is_done():
        await interaction.followup.send(msg, **kwargs)
    else:
        await interaction.response.send_message(msg, **kwargs)


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
    name: str
    creator_id: int
    deadline: datetime.datetime
    task: LeaderboardTask
    gpu_types: List[str]


class LeaderboardRankedEntry(TypedDict):
    rank: int
    submission_name: str
    submission_time: datetime.datetime
    submission_score: float
    leaderboard_name: str
    user_id: int
    gpu_type: str


def build_task_config(
    task: LeaderboardTask = None,
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

        if task.lang == Language.Python:
            return {
                "lang": task.lang.value,
                "arch": arch,
                "main": task.config.main,
                "sources": all_files,
                "benchmarks": task.benchmarks,
                "tests": task.tests,
                "mode": mode.value,
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
                "lang": task.lang.value,
                "arch": arch,
                "sources": sources,
                "headers": headers,
                "tests": task.tests,
                "benchmarks": task.benchmarks,
                "include_dirs": task.config.include_dirs,
                "mode": mode.value,
            }
