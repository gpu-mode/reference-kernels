from enum import Enum
from typing import Optional, Tuple, Type

import discord
from discord import app_commands
from discord.ext import commands
from report import generate_report
from run_eval import FullResult
from task import LeaderboardTask
from utils import build_task_config, send_discord_message, setup_logging

logger = setup_logging()


class ProgressReporter:
    def __init__(self, status_msg: discord.Message, header: str):
        self.header = header
        self.lines = []
        self.status = status_msg

    @staticmethod
    async def make_reporter(thread: discord.Thread, content: str):
        status_msg = await thread.send(f"**{content}**\n")
        return ProgressReporter(status_msg, content)

    async def push(self, content: str):
        self.lines.append(f"> {content}")
        await self._update_message()

    async def update(self, new_content: str):
        self.lines[-1] = f"> {new_content}"
        await self._update_message()

    async def update_header(self, new_header):
        self.header = new_header
        await self._update_message()

    async def _update_message(self):
        message = str.join("\n", [f"**{self.header}**"] + self.lines)
        await self.status.edit(content=message, suppress=True)


class SubmitCog(commands.Cog):
    """
    Base class for code submission / run schedular cogs.

    Derived classes need to implement a `_get_arch(self, gpu_type: app_commands.Choice[str])`
    method to translate the selected GPU to an architecture argument for Cuda,
    and a
    ```
    run_submission(self, config: dict, gpu_type: GPUType,
        status: ProgressReporter) -> FullResult
    ```
    coroutine, which handles the actual submission.

    This base class will register a `run` subcommand with the runner's name, which can be used
    to run a single (non-leaderboard) script.
    """

    def __init__(self, bot, name: str, gpus: Type[Enum]):
        self.bot = bot
        self.name = name

        choices = [app_commands.Choice(name=c.name, value=c.value) for c in gpus]

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
            gpu_type=f"Choose the GPU type for {name}",
        )(run)

        self.run_script = bot.run_group.command(
            name=self.name.lower(), description=f"Run a script using {self.name}"
        )(run)

    async def submit_leaderboard(
        self,
        interaction: discord.Interaction,
        script: discord.Attachment,
        gpu_type: app_commands.Choice[str],
        task: LeaderboardTask,
    ) -> Tuple[Optional[discord.Thread], Optional[FullResult]]:
        """
        Function invoked by `leaderboard_cog` to handle a leaderboard run.
        """
        thread, result = await self._handle_submission(
            interaction,
            gpu_type,
            script=script,
            task=task,
        )

        return thread, result

    async def run_script(
        self,
        interaction: discord.Interaction,
        script: discord.Attachment,
        gpu_type: app_commands.Choice[str],
    ):
        """
        Function invoked by the `run` command to run a single script.
        """
        await self._handle_submission(
            interaction,
            gpu_type,
            script=script,
            task=None,
        )

    async def _handle_submission(
        self,
        interaction: discord.Interaction,
        gpu_type: app_commands.Choice[str],
        script: discord.Attachment,
        task: Optional[LeaderboardTask],
    ) -> Tuple[Optional[discord.Thread], Optional[FullResult]]:
        """
        Generic function to handle code submissions.
        Args:
            interaction: Interaction that started this command.
            gpu_type: Which GPU to run on.
            script: File that contains the submitted script.
            task: Task specification, of provided

        Returns:
            if successful, returns the created discord thread, and the result of
            the run.
        """

        script_content = await self._validate_input_file(interaction, script)
        if script_content is None:
            return None, None

        # TODO figure out the correct way to handle messaging here
        thread = await self.bot.create_thread(interaction, gpu_type.name, f"{self.name} Job")
        await thread.send(
            f"Starting {self.name} job for " f"`{script.filename}` with {gpu_type.name}..."
        )

        status = await ProgressReporter.make_reporter(thread, f"Running on {self.name}...")

        config = build_task_config(
            task=task,
            submission_content=script_content,
            arch=self._get_arch(gpu_type),
        )

        logger.info("submitting task %s to runner %s", config, self.name)

        result = await self._run_submission(config, gpu_type, status)
        await status.update_header(f"Running on {self.name}... ✅ success")
        await generate_report(thread, result, has_tests=task is not None)

        return thread, result

    async def _validate_input_file(
        self,
        interaction: discord.Interaction,
        script: discord.Attachment,
    ) -> Optional[str]:
        # check file extension
        if not script.filename.endswith((".py", ".cu", ".cuh", ".cpp")):
            await send_discord_message(
                interaction,
                "Please provide a Python (.py) or CUDA (.cu / .cuh / .cpp) file",
                ephemeral=True,
            )
            return None

        #  load and decode
        try:
            return (await script.read()).decode("utf-8")
        except UnicodeError:
            await send_discord_message(
                interaction,
                f"Could not decode your file `{script.filename}`.\n" f"Is it UTF-8?",
                ephemeral=True,
            )
            return None

    async def _run_submission(
        self, config: dict, gpu_type: app_commands.Choice[str], status: ProgressReporter
    ) -> FullResult:
        """
        Run a submission specified by `config`.
        To be implemented in derived classes.
        Args:
            config: the config object containing all necessary runner information.
            gpu_type: Which GPU to run for.
            status: callback object that allows updating the status message in discord

        Returns:
            Result of running `config`.
        """
        raise NotImplementedError()

    def _get_arch(self, gpu_type: app_commands.Choice[str]):
        raise NotImplementedError()
