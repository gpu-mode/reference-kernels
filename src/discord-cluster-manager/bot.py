import discord
from discord import app_commands
from discord.ext import commands
import argparse
from utils import setup_logging
from cogs.misc_cog import BotManagerCog
from datetime import datetime
from consts import (
    init_environment,
    DISCORD_TOKEN,
    DISCORD_DEBUG_TOKEN,
    DISCORD_CLUSTER_STAGING_ID,
    DISCORD_DEBUG_CLUSTER_STAGING_ID,
)
from cogs.modal_cog import ModalCog
from cogs.github_cog import GitHubCog
from cogs.leaderboard_cog import LeaderboardCog

logger = setup_logging()


class ClusterBot(commands.Bot):
    def __init__(self, debug_mode=False):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents, command_prefix="!")
        self.debug_mode = debug_mode

        # Create the run group
        self.run_group = app_commands.Group(
            name="run", description="Run jobs on different platforms"
        )
        self.tree.add_command(self.run_group)

        self.leaderboard_group = app_commands.Group(
            name="leaderboard", description="Leaderboard commands"
        )
        self.tree.add_command(self.leaderboard_group)

    async def setup_hook(self):
        logger.info(f"Syncing commands for staging guild {DISCORD_CLUSTER_STAGING_ID}")
        try:
            # Load cogs
            await self.add_cog(ModalCog(self))
            await self.add_cog(GitHubCog(self))
            await self.add_cog(BotManagerCog(self))
            await self.add_cog(LeaderboardCog(self))

            guild_id = (
                DISCORD_CLUSTER_STAGING_ID
                if not self.debug_mode
                else DISCORD_DEBUG_CLUSTER_STAGING_ID
            )

            if guild_id:
                guild = discord.Object(id=int(guild_id))
                self.tree.clear_commands(guild=guild)
                logger.info("Cleared existing commands")
                self.tree.copy_global_to(guild=guild)
                await self.tree.sync(guild=guild)
                commands = await self.tree.fetch_commands(guild=guild)
                logger.info(f"Synced commands: {[cmd.name for cmd in commands]}")
        except Exception as e:
            logger.error(f"Failed to sync commands: {e}")

    async def on_ready(self):
        logger.info(f"Logged in as {self.user}")
        for guild in self.guilds:
            try:
                if self.debug_mode:
                    await guild.me.edit(nick="Cluster Bot (Staging)")
                else:
                    await guild.me.edit(nick="Cluster Bot")
            except Exception as e:
                logger.warning(f"Failed to update nickname in guild {guild.name}: {e}")

    async def create_thread(
        self, interaction: discord.Interaction, gpu_name: str, job_name: str
    ) -> discord.Thread:
        thread = await interaction.channel.create_thread(
            name=f"{job_name} ({gpu_name}) - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            type=discord.ChannelType.public_thread,
            auto_archive_duration=1440,
        )
        return thread

    async def send_chunked_message(
        self, channel, content: str, code_block: bool = True
    ):
        """
        Send a long message in chunks to avoid Discord's message length limit

        Args:
            channel: The discord channel/thread to send to
            content: The content to send
            code_block: Whether to wrap the content in code blocks
        """
        chunk_size = 1900  # Leave room for code block syntax
        chunks = [
            content[i : i + chunk_size] for i in range(0, len(content), chunk_size)
        ]

        for i, chunk in enumerate(chunks):
            if code_block:
                await channel.send(
                    f"```\nOutput (part {i+1}/{len(chunks)}):\n{chunk}\n```"
                )
            else:
                await channel.send(chunk)


def main():
    init_environment()

    parser = argparse.ArgumentParser(description="Run the Discord Cluster Bot")
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug/staging mode"
    )
    args = parser.parse_args()

    logger.info("Starting bot...")
    token = DISCORD_DEBUG_TOKEN if args.debug else DISCORD_TOKEN

    if args.debug and not token:
        raise ValueError("DISCORD_DEBUG_TOKEN not found")

    client = ClusterBot(debug_mode=args.debug)
    client.run(token)


if __name__ == "__main__":
    main()
