import functools
import logging

import discord
from utils import KernelBotError, limit_length, setup_logging

logger = setup_logging(__name__)


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


async def leaderboard_name_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> list[discord.app_commands.Choice[str]]:
    """Return leaderboard names that match the current typed name"""
    try:
        bot = interaction.client
        name_cache = bot.leaderboard_db.name_cache
        cached_value = name_cache[current]
        if cached_value is not None:
            return cached_value

        with bot.leaderboard_db as db:
            leaderboards = db.get_leaderboard_names(active_only=True)
        filtered = [lb for lb in leaderboards if current.lower() in lb.lower()]
        name_cache[current] = [
            discord.app_commands.Choice(name=name, value=name) for name in filtered[:25]
        ]
        return name_cache[current]
    except Exception as e:
        logger.exception("Error in leaderboard autocomplete", exc_info=e)
        return []


async def _send_split_log(thread: discord.Thread, partial_message: str, header: str, log: str):
    if len(partial_message) + len(log) + len(header) < 1900:
        partial_message += f"\n\n## {header}:\n"
        partial_message += f"```\n{log}```"
        return partial_message
    else:
        # send previous chunk
        if len(partial_message) > 0:
            await thread.send(partial_message)
        lines = log.splitlines()
        chunks = []
        partial_message = ""
        for line in lines:
            if len(partial_message) + len(line) < 1900:
                partial_message += line + "\n"
            else:
                if partial_message != "":
                    chunks.append(partial_message)
                partial_message = line

        if partial_message != "":
            chunks.append(partial_message)

        # now, format the chunks
        for i, chunk in enumerate(chunks):
            partial_message = f"\n\n## {header} ({i+1}/{len(chunks)}):\n"
            partial_message += f"```\n{limit_length(chunk, 1900)}```"
            await thread.send(partial_message)

        return ""
