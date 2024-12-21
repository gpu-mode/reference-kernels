import asyncio
import re
from unittest.mock import AsyncMock

import discord
from cogs.github_cog import GitHubCog
from cogs.modal_cog import ModalCog
from discord import app_commands
from discord.ext import commands
from utils import send_discord_message, setup_logging

logger = setup_logging()


def create_mock_attachment():
    "Create an AsyncMock to simulate discord.Attachment"

    mock_attachment = AsyncMock(spec=discord.Attachment)
    mock_attachment.filename = "test_script.py"
    mock_attachment.content_type = "text/plain"
    mock_attachment.read = AsyncMock(
        return_value="print('Hello, world!')".encode("utf-8")
    )
    return mock_attachment


script_file = create_mock_attachment()


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
    ) -> bool:
        github_command = github_cog.run_github
        github_thread = await github_command.callback(
            github_cog, interaction, script_file, choice
        )

        message_contents = [
            msg.content async for msg in github_thread.history(limit=None)
        ]

        required_patterns = [
            "Processing `.*` with",
            "GitHub Action triggered! Run ID:",
            "Training completed with status: success",
            ".*```\nLogs.*:",
            "View the full run at:",
        ]

        all_patterns_found = all(
            any(
                re.search(pattern, content, re.DOTALL) is not None
                for content in message_contents
            )
            for pattern in required_patterns
        )

        if all_patterns_found:
            await send_discord_message(
                interaction,
                f"✅ GitHub run ({choice.name}) completed successfully - "
                "all expected messages found!",
            )
            return True
        else:
            missing_patterns = [
                pattern
                for pattern in required_patterns
                if not any(
                    re.search(pattern, content, re.DOTALL)
                    for content in message_contents
                )
            ]
            await send_discord_message(
                interaction,
                f"❌ GitHub run ({choice.name}) verification failed. Missing expected messages:\n"
                + "\n".join(f"- {pattern}" for pattern in missing_patterns),
            )
            return False

    async def verify_modal_run(
        self, modal_cog: ModalCog, interaction: discord.Interaction
    ) -> bool:
        t4 = app_commands.Choice(name="NVIDIA T4", value="t4")
        modal_command = modal_cog.run_modal

        modal_thread = await modal_command.callback(
            modal_cog, interaction, script_file, t4
        )

        message_contents = [
            msg.content async for msg in modal_thread.history(limit=None)
        ]

        required_patterns = [
            "Processing `.*` with",
            "Running on Modal...",
            "Modal execution result:",
        ]

        all_patterns_found = all(
            any(
                re.search(pattern, content, re.DOTALL) is not None
                for content in message_contents
            )
            for pattern in required_patterns
        )

        if all_patterns_found:
            await send_discord_message(
                interaction,
                "✅ Modal run completed successfully - all expected messages found!",
            )
            return True
        else:
            missing_patterns = [
                pattern
                for pattern in required_patterns
                if not any(
                    re.search(pattern, content, re.DOTALL)
                    for content in message_contents
                )
            ]
            await send_discord_message(
                interaction,
                "❌ Modal run verification failed. Missing expected messages:\n"
                + "\n".join(f"- {pattern}" for pattern in missing_patterns),
            )
            return False

    @app_commands.command(name="verifyruns")
    async def verify_runs(self, interaction: discord.Interaction):
        """Verify runs on on Modal, GitHub Nvidia, and GitHub AMD."""

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
                self.verify_github_run(github_cog, nvidia, interaction),
                self.verify_github_run(github_cog, amd, interaction),
                self.verify_modal_run(modal_cog, interaction),
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
