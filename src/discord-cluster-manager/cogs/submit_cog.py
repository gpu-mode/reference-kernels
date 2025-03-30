from typing import TYPE_CHECKING, Optional

from launchers import Launcher

if TYPE_CHECKING:
    from bot import ClusterBot

import discord
from consts import GPU, GPU_TO_SM, SubmissionMode, get_gpu_by_name
from discord import app_commands
from discord.ext import commands
from report import MultiProgressReporter, RunProgressReporter, generate_report, make_short_report
from run_eval import FullResult
from task import LeaderboardTask
from utils import build_task_config, send_discord_message, setup_logging, with_error_handling

logger = setup_logging()


class SubmitCog(commands.Cog):
    """
    Code submission / run schedular cogs.

    Actual submission logic is handled by the launcher object.
    """

    def __init__(self, bot, launcher: Launcher):
        self.bot: ClusterBot = bot
        self.launcher = launcher
        self.name = launcher.name

        choices = [app_commands.Choice(name=c.name, value=c.value) for c in launcher.gpus]

        run_fn = self.run_script

        # note: these helpers want to set custom attributes on the function, but `method`
        # does not allow setting any attributes, so we define this wrapper
        async def run(
            interaction: discord.Interaction,
            script: discord.Attachment,
            gpu_type: app_commands.Choice[str],
        ):
            return await run_fn(interaction, script, gpu_type)

        run = app_commands.choices(gpu_type=choices)(run)
        run = app_commands.describe(
            script="The Python/CUDA script file to run",
            gpu_type=f"Choose the GPU type for {launcher.name}",
        )(run)

        # For now, direct (non-leaderboard) submissions are debug-only.
        if self.bot.debug_mode:
            self.run_script = bot.run_group.command(
                name=self.name.lower(), description=f"Run a script using {self.name}"
            )(run)

    async def submit_leaderboard(
        self,
        interaction: discord.Interaction,
        script: discord.Attachment,
        gpu_type: GPU,
        reporter: RunProgressReporter,
        task: LeaderboardTask,
        mode: SubmissionMode,
    ) -> Optional[FullResult]:
        """
        Function invoked by `leaderboard_cog` to handle a leaderboard run.
        """
        result = await self._handle_submission(
            interaction,
            gpu_type,
            reporter,
            script=script,
            task=task,
            mode=mode,
        )

        return result

    @with_error_handling
    async def run_script(
        self,
        interaction: discord.Interaction,
        script: discord.Attachment,
        gpu_type: app_commands.Choice[str],
    ):
        """
        Function invoked by the `run` command to run a single script.
        """
        reporter = MultiProgressReporter("Script run")
        rep = reporter.add_run(f"{gpu_type.name}")
        await reporter.show(interaction)
        gpu_type = get_gpu_by_name(gpu_type.name)
        await self._handle_submission(
            interaction, gpu_type, rep, script=script, task=None, mode=SubmissionMode.SCRIPT
        )

    async def _handle_submission(
        self,
        interaction: discord.Interaction,
        gpu_type: GPU,
        reporter: RunProgressReporter,
        script: discord.Attachment,
        task: Optional[LeaderboardTask],
        mode: SubmissionMode,
    ) -> Optional[FullResult]:
        """
        Generic function to handle code submissions.
        Args:
            interaction: Interaction that started this command.
            gpu_type: Which GPU to run on.
            script: File that contains the submitted script.
            task: Task specification, of provided

        Returns:
            if successful, returns the result of the run.
        """
        script_content = await self._validate_input_file(interaction, script)
        if script_content is None:
            return None

        # TODO figure out the correct way to handle messaging here
        if mode != SubmissionMode.PRIVATE:
            thread = await interaction.channel.create_thread(
                name=f"{script.filename} on {gpu_type.name} ({self.name})",
                type=discord.ChannelType.private_thread,
                auto_archive_duration=1440,
            )
            await thread.add_user(interaction.user)
        config = build_task_config(
            task=task, submission_content=script_content, arch=self._get_arch(gpu_type), mode=mode
        )

        logger.info("submitting task to runner %s", self.name)

        result = await self.launcher.run_submission(config, gpu_type, reporter)

        if not result.success:
            await reporter.update_title(reporter.title + " ❌ failure")
            await reporter.push(result.error)
            return result
        else:
            await reporter.update_title(reporter.title + " ✅ success")

        await reporter.push(make_short_report(
            result.runs,
            full=mode in [SubmissionMode.PRIVATE, SubmissionMode.LEADERBOARD])
        )
        if mode != SubmissionMode.PRIVATE:
            try:
                await generate_report(thread, result.runs)
                await reporter.push(f"See results at {thread.jump_url}")
            except Exception as E:
                logger.error("Error generating report. Result: %s", result, exc_info=E)
                raise

        return result

    async def _validate_input_file(
        self,
        interaction: discord.Interaction,
        script: discord.Attachment,
    ) -> Optional[str]:
        # load and decode
        try:
            return (await script.read()).decode("utf-8")
        except UnicodeError:
            await send_discord_message(
                interaction,
                f"Could not decode your file `{script.filename}`.\nIs it UTF-8?",
                ephemeral=True,
            )
            return None

    def _get_arch(self, gpu_type: GPU):
        return GPU_TO_SM[gpu_type.name]
