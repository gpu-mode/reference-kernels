from typing import TYPE_CHECKING, Optional

from launchers import Launcher

if TYPE_CHECKING:
    from bot import ClusterBot

import discord
from consts import SubmissionMode, get_gpu_by_name
from discord import app_commands
from discord.ext import commands
from discord_reporter import MultiProgressReporter
from discord_utils import send_discord_message, with_error_handling
from utils import (
    setup_logging,
)

logger = setup_logging()


class SubmitCog(commands.Cog):
    """
    Code submission / run schedular cogs.

    Actual submission logic is handled by the launcher object.
    """

    def __init__(self, bot):
        self.bot: ClusterBot = bot
        handled_launchers = set()
        for launcher in self.bot.backend.launcher_map.values():
            if id(launcher) in handled_launchers:
                continue
            handled_launchers.add(id(launcher))
            self.register_launcher(launcher)

    def register_launcher(self, launcher: Launcher):
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
            self.bot.run_group.command(
                name=launcher.name.lower(), description=f"Run a script using {launcher.name}"
            )(run)

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
        reporter = MultiProgressReporter(interaction, "Script run")
        rep = reporter.add_run(f"{gpu_type.name}")
        await reporter.show()
        gpu_type = get_gpu_by_name(gpu_type.name)
        script_content = await self._validate_input_file(interaction, script)
        if script_content is None:
            return

        await self.bot.backend.handle_submission(
            gpu_type,
            rep,
            code=script_content,
            name=script.filename,
            task=None,
            mode=SubmissionMode.SCRIPT,
        )

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
