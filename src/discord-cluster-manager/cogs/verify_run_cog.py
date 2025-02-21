import asyncio
import datetime
import re
import uuid
from pathlib import Path
from unittest.mock import AsyncMock

import discord
from cogs.github_cog import GitHubCog
from cogs.leaderboard_cog import LeaderboardSubmitCog
from cogs.modal_cog import ModalCog
from consts import SubmissionMode
from discord import app_commands
from discord.ext import commands
from task import make_task
from utils import send_discord_message, setup_logging

logger = setup_logging()


def create_mock_attachment(file_name: str, content: str):
    "Create an AsyncMock to simulate discord.Attachment"

    mock_attachment = AsyncMock(spec=discord.Attachment)
    mock_attachment.filename = file_name
    mock_attachment.content_type = "text/plain"
    mock_attachment.read = AsyncMock(return_value=content.encode("utf-8"))
    return mock_attachment


class VerifyRunCog(commands.Cog):
    """
    A Discord cog for verifying the success of training runs.

    A cog that verifies training runs across different platforms and GPU types.
    Runs test scripts on GitHub (NVIDIA and AMD) and Modal to validate that the
    runs complete successfully. Each run is monitored for expected output
    messages.
    """

    def __init__(self, bot):
        self.bot = bot

    async def verify_github_run(
        self,
        github_cog: GitHubCog,
        choice: app_commands.Choice,
        interaction: discord.Interaction,
        lang: str,
    ) -> bool:
        # async_submit_cog_job
        github_command = github_cog.submit_leaderboard
        if lang == "py":
            sub_code = create_mock_attachment(
                "submission.py", Path("examples/softmax_py/submission.py").read_text()
            )
            task = make_task("examples/softmax_py")
        else:
            sub_code = create_mock_attachment(
                "test.cu", Path("examples/identity_cuda/submission.cu").read_text()
            )
            task = make_task("examples/identity_cuda")

        github_thread, _ = await github_command(
            interaction, sub_code, choice, task=task, mode=SubmissionMode.TEST
        )

        message_contents = [msg.content async for msg in github_thread.history(limit=None)]

        required_patterns = ["Running on GitHub...", "Passed 5/5 tests"]

        all_patterns_found = all(
            any(re.search(pattern, content, re.DOTALL) is not None for content in message_contents)
            for pattern in required_patterns
        )

        if all_patterns_found:
            await send_discord_message(
                interaction,
                f"✅ GitHub run ({choice.name}) for {lang} completed successfully - "
                "all expected messages found!",
            )
            return True
        else:
            missing_patterns = [
                pattern
                for pattern in required_patterns
                if not any(re.search(pattern, content, re.DOTALL) for content in message_contents)
            ]
            await send_discord_message(
                interaction,
                f"❌ GitHub run ({choice.name}) for {lang} verification failed. "
                + "Missing expected messages:\n"
                + "\n".join(f"- {pattern}" for pattern in missing_patterns),
            )
            return False

    async def verify_modal_run(
        self, modal_cog: ModalCog, interaction: discord.Interaction, lang: str
    ) -> bool:
        t4 = app_commands.Choice(name="T4", value="t4")
        modal_command = modal_cog.submit_leaderboard

        if lang == "py":
            sub_code = create_mock_attachment(
                "submission.py", Path("examples/identity_py/submission.py").read_text()
            )
            task = make_task("examples/identity_py")
        else:
            sub_code = create_mock_attachment(
                "test.cu", Path("examples/identity_cuda/submission.cu").read_text()
            )
            task = make_task("examples/identity_cuda")

        modal_thread, _ = await modal_command(
            interaction, sub_code, t4, task=task, mode=SubmissionMode.TEST
        )

        message_contents = [msg.content async for msg in modal_thread.history(limit=None)]

        required_patterns = ["Running on Modal...", "Passed 5/5 tests"]

        all_patterns_found = all(
            any(re.search(pattern, content, re.DOTALL) is not None for content in message_contents)
            for pattern in required_patterns
        )

        if all_patterns_found:
            await send_discord_message(
                interaction,
                f"✅ Modal run for {lang} completed successfully - all expected messages found!",
            )
            return True
        else:
            missing_patterns = [
                pattern
                for pattern in required_patterns
                if not any(re.search(pattern, content, re.DOTALL) for content in message_contents)
            ]
            await send_discord_message(
                interaction,
                f"❌ Modal run verification for {lang} failed. Missing expected messages:\n"
                + "\n".join(f"- {pattern}" for pattern in missing_patterns),
            )
            return False

    @app_commands.command(name="verify-task")
    async def verify_task(self, interaction: discord.Interaction, task: str):
        directory = Path("examples") / task
        if not directory.resolve().is_relative_to(Path.cwd() / "examples"):
            await send_discord_message(interaction, f"Invalid path {directory.resolve()}")
            return
        try:
            task = make_task(directory)
        except Exception as E:
            logger.exception("Could not make task", exc_info=E)
            await send_discord_message(interaction, f"Invalid task {directory}")
            return
        await send_discord_message(interaction, f"Testing {directory}")

        lb_name = f"test.{uuid.uuid4().hex}"
        # create the dummy leaderboard
        with self.bot.leaderboard_db as db:  # type: LeaderboardDB
            err = db.create_leaderboard(
                {
                    "name": lb_name,
                    "deadline": datetime.datetime.now() + datetime.timedelta(days=1),
                    "task": task,
                    "gpu_types": "T4",
                    "creator_id": interaction.user.id,
                }
            )
            if err:
                logger.error(err)
                await send_discord_message(interaction, err)

        try:
            # make submissions
            submissions = []
            for sub in directory.glob("submission*"):
                for mode in [
                    SubmissionMode.TEST,
                    SubmissionMode.BENCHMARK,
                    SubmissionMode.LEADERBOARD,
                ]:
                    submissions.append(self.verify_submission(interaction, lb_name, sub, mode))
            await asyncio.gather(*submissions)
        except Exception as E:
            logger.exception("Error in LB test", exc_info=E)
        finally:
            with self.bot.leaderboard_db as db:
                db.delete_leaderboard(lb_name)

        await send_discord_message(interaction, "Done")

    async def verify_submission(
        self, interaction: discord.Interaction, lb_name: str, sub: Path, mode: SubmissionMode
    ):
        lb_cog = LeaderboardSubmitCog(self.bot)
        script = create_mock_attachment(sub.name, sub.read_text())
        await lb_cog.on_submit_hook(interaction, lb_name, script, mode, cmd_gpus=["T4"])

    @app_commands.command(name="verifyruns")
    async def verify_runs(self, interaction: discord.Interaction):
        """Verify runs on Modal, GitHub Nvidia, and GitHub AMD."""

        try:
            if not interaction.response.is_done():
                await interaction.response.defer()

            modal_cog = self.bot.get_cog("ModalCog")
            github_cog = self.bot.get_cog("GitHubCog")

            if not all([modal_cog, github_cog]):
                await send_discord_message(interaction, "❌ Required cogs not found!")
                return

            nvidia = app_commands.Choice(name="NVIDIA", value="nvidia")
            amd = app_commands.Choice(name="AMD", value="amd")

            results = await asyncio.gather(
                self.verify_github_run(github_cog, nvidia, interaction, "py"),
                self.verify_github_run(github_cog, nvidia, interaction, "cu"),
                self.verify_modal_run(modal_cog, interaction, "py"),
                self.verify_github_run(github_cog, amd, interaction, "py"),
                self.verify_modal_run(modal_cog, interaction, "cu"),
            )

            if all(results):
                await send_discord_message(interaction, "✅ All runs completed successfully!")
            else:
                await send_discord_message(
                    interaction,
                    "❌ Some runs failed! Consult messages above for details.",
                )

        except Exception as e:
            logger.error(f"Error starting verification runs: {e}", exc_info=True)
            await send_discord_message(
                interaction, f"❌ Problem performing verification runs: {str(e)}"
            )
