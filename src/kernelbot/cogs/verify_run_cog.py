# ruff: noqa: E501
import asyncio
import datetime
import uuid
from pathlib import Path
from unittest.mock import AsyncMock

import discord
from discord import app_commands
from discord.app_commands import Choice
from discord.ext import commands

from kernelbot.cogs import admin_cog
from kernelbot.cogs.leaderboard_cog import LeaderboardSubmitCog
from kernelbot.discord_reporter import MultiProgressReporterDiscord
from kernelbot.discord_utils import send_discord_message, with_error_handling
from kernelbot.env import env
from libkernelbot.consts import GPU, SubmissionMode, get_gpu_by_name
from libkernelbot.leaderboard_db import RunItem, SubmissionItem
from libkernelbot.task import make_task_definition
from libkernelbot.utils import setup_logging

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

    async def trigger_run(self, interaction: discord.Interaction, gpu: GPU, reporter, lang: str):
        submit_leaderboard = self.bot.backend.submit_leaderboard

        if lang == "py":
            sub_code = create_mock_attachment(
                "submission.py", Path("examples/identity_py/submission.py").read_text()
            )
            leaderboard = make_task_definition("examples/identity_py")
        else:
            sub_code = create_mock_attachment(
                "test.cu", Path("examples/identity_cuda/submission.cu").read_text()
            )
            leaderboard = make_task_definition("examples/identity_cuda")

        return await submit_leaderboard(
            interaction,
            -1,
            sub_code,
            gpu,
            reporter=reporter,
            task=leaderboard.task,
            mode=SubmissionMode.TEST,
            seed=None,
        )

    async def verify_github_run(
        self,
        interaction: discord.Interaction,
        gpu: GPU,
        reporter,
        lang: str,
    ) -> bool:
        result = await self.trigger_run(interaction, gpu, reporter, lang)
        return result.success
        #
        # message_contents = [msg.content async for msg in github_thread.history(limit=None)]
        #
        # required_patterns = ["Running on GitHub...", "Passed 5/5 tests"]
        #
        # all_patterns_found = all(
        #     any(re.search(pattern, content, re.DOTALL) is not None for content in message_contents)
        #     for pattern in required_patterns
        # )
        #
        # if all_patterns_found:
        #     await send_discord_message(
        #         interaction,
        #         f"✅ GitHub run ({gpu.name}) for {lang} completed successfully - "
        #         "all expected messages found!",
        #     )
        #     return True
        # else:
        #     missing_patterns = [
        #         pattern
        #         for pattern in required_patterns
        #         if not any(re.search(pattern, content, re.DOTALL) for content in message_contents)
        #     ]
        #     await send_discord_message(
        #         interaction,
        #         f"❌ GitHub run ({gu.name}) for {lang} verification failed. "
        #         + "Missing expected messages:\n"
        #         + "\n".join(f"- {pattern}" for pattern in missing_patterns),
        #     )
        #     return False

    async def verify_modal_run(
        self,
        interaction: discord.Interaction,
        gpu: GPU,
        reporter,
        lang: str,
    ) -> bool:
        result = await self.trigger_run(interaction, gpu, reporter, lang)
        return result.success
        # t4 = ModalGPU.T4
        # modal_command = modal_cog.submit_leaderboard
        #
        # if lang == "py":
        #     sub_code = create_mock_attachment(
        #         "submission.py", Path("examples/identity_py/submission.py").read_text()
        #     )
        #     task = make_task("examples/identity_py")
        # else:
        #     sub_code = create_mock_attachment(
        #         "test.cu", Path("examples/identity_cuda/submission.cu").read_text()
        #     )
        #     task = make_task("examples/identity_cuda")
        #
        # modal_thread, _ = await modal_command(
        #     interaction, -1, sub_code, t4, reporter=MockReporter(), task=task, mode=SubmissionMode.TEST, seed=None
        # )
        #
        # message_contents = [msg.content async for msg in modal_thread.history(limit=None)]
        #
        # required_patterns = ["Running on Modal...", "Passed 5/5 tests"]
        #
        # all_patterns_found = all(
        #     any(re.search(pattern, content, re.DOTALL) is not None for content in message_contents)
        #     for pattern in required_patterns
        # )
        #
        # if all_patterns_found:
        #     await send_discord_message(
        #         interaction,
        #         f"✅ Modal run for {lang} completed successfully - all expected messages found!",
        #     )
        #     return True
        # else:
        #     missing_patterns = [
        #         pattern
        #         for pattern in required_patterns
        #         if not any(re.search(pattern, content, re.DOTALL) for content in message_contents)
        #     ]
        #     await send_discord_message(
        #         interaction,
        #         f"❌ Modal run verification for {lang} failed. Missing expected messages:\n"
        #         + "\n".join(f"- {pattern}" for pattern in missing_patterns),
        #     )
        #     return False

    @app_commands.command(name="verify-task")
    @app_commands.autocomplete(task=admin_cog.leaderboard_dir_autocomplete)
    @app_commands.choices(
        mode=[
            Choice(name=SubmissionMode.TEST.name, value=SubmissionMode.TEST.value),
            Choice(name=SubmissionMode.BENCHMARK.name, value=SubmissionMode.BENCHMARK.value),
            Choice(name=SubmissionMode.LEADERBOARD.name, value=SubmissionMode.LEADERBOARD.value),
            Choice(name="All", value="all"),
        ]
    )
    @with_error_handling
    async def verify_task(
        self, interaction: discord.Interaction, task: str, mode: Choice[str] = None
    ):
        directory = Path(env.PROBLEM_DEV_DIR) / task
        if not directory.resolve().is_relative_to(Path.cwd() / env.PROBLEM_DEV_DIR):
            await send_discord_message(interaction, f"Invalid path {directory.resolve()}")
            return
        try:
            task = make_task_definition(directory)
        except Exception as E:
            logger.exception("Could not make task", exc_info=E)
            await send_discord_message(interaction, f"Invalid task {directory}")
            return
        await send_discord_message(interaction, f"Testing {directory}")

        modes = []
        if mode is None:
            modes = [SubmissionMode.LEADERBOARD]
        elif mode.value == "all":
            modes = [SubmissionMode.TEST, SubmissionMode.BENCHMARK, SubmissionMode.LEADERBOARD]
        else:
            modes = [SubmissionMode(mode.value)]

        lb_name = f"test.{uuid.uuid4().hex}"
        # create the dummy leaderboard
        with self.bot.leaderboard_db as db:  # type: LeaderboardDB
            db.create_leaderboard(
                {
                    "name": lb_name,
                    "deadline": datetime.datetime.now() + datetime.timedelta(days=1),
                    "task": task,
                    "gpu_types": "T4",
                    "creator_id": interaction.user.id,
                }
            )
        try:
            # make submissions
            submissions = []
            reports = []
            for sub in directory.glob("solutions/*/*"):
                for mode in modes:
                    submissions.append(
                        self.verify_submission(interaction, lb_name, sub, mode, reports)
                    )
            await asyncio.gather(*submissions)
        except Exception as E:
            logger.exception("Error in LB test", exc_info=E)
            await send_discord_message(interaction, str(E), ephemeral=True)
            return
        finally:
            with self.bot.leaderboard_db as db:
                db.delete_leaderboard(lb_name, force=True)

        report = f"Report:```{'\n'.join(sorted(reports))}```"

        await send_discord_message(interaction, report)

    async def verify_submission(  # noqa: C901
        self,
        interaction: discord.Interaction,
        lb_name: str,
        sub: Path,
        mode: SubmissionMode,
        reports: list[str],
    ):
        lb_cog = LeaderboardSubmitCog(self.bot)
        script = create_mock_attachment(sub.name, sub.read_text())
        sub_id = await lb_cog.on_submit_hook(interaction, lb_name, script, mode, cmd_gpus=["T4"])

        run_id = f"{sub.parent.name}/{sub.name}:"

        if sub_id == -1:
            reports.append(f"❌ {run_id:20} submitting failed")
            return

        report_success = True

        # verify results
        with self.bot.leaderboard_db as db:
            sub_data: SubmissionItem = db.get_submission_by_id(sub_id)
        if sub_data is None:
            reports.append(f"❌ {run_id:20} cannot find in db")
            return

        if sub_data["done"] is not True:
            reports.append(f"❌ {run_id:20} is unfinished")
            return

        if sub.parent.name == "correct":
            run: RunItem
            for run in sub_data["runs"]:
                if run["passed"] is not True:
                    reports.append(f"❌ {run_id:20} run {run['mode']} failed")
                    report_success = False
        elif sub.parent.name == "wrong":
            for run in sub_data["runs"]:
                if run["passed"] is True:
                    reports.append(f"❌ {run_id:20} run {run['mode']} passed")
                    report_success = False

        if report_success:
            reports.append(f"✅ {run_id:20} {mode.name} behaved as expected")

    @app_commands.command(name="verifyruns")
    async def verify_runs(self, interaction: discord.Interaction):
        """Verify runs on Modal, GitHub Nvidia, and GitHub AMD."""

        try:
            if not interaction.response.is_done():
                await interaction.response.defer()

            nvidia = get_gpu_by_name("nvidia")
            amd = get_gpu_by_name("mi300")
            t4 = get_gpu_by_name("T4")

            reporter = MultiProgressReporterDiscord(interaction)
            await reporter.show("Verifying")

            results = await asyncio.gather(
                self.verify_github_run(interaction, nvidia, reporter.add_run("NVIDIA-PY"), "py"),
                self.verify_github_run(interaction, nvidia, reporter.add_run("NVIDIA-CU"), "cu"),
                self.verify_modal_run(interaction, t4, reporter.add_run("MODAL-PY"), "py"),
                self.verify_github_run(interaction, amd, reporter.add_run("AMD-PY"), "py"),
                self.verify_modal_run(interaction, t4, reporter.add_run("MODAL-CU"), "py"),
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
