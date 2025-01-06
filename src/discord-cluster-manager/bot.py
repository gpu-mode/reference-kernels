import argparse
from datetime import datetime

import discord
from cogs.github_cog import GitHubCog
from cogs.leaderboard_cog import LeaderboardCog
from cogs.misc_cog import BotManagerCog
from cogs.modal_cog import ModalCog
from cogs.verify_run_cog import VerifyRunCog
from consts import (
    DISCORD_CLUSTER_STAGING_ID,
    DISCORD_DEBUG_CLUSTER_STAGING_ID,
    DISCORD_DEBUG_TOKEN,
    DISCORD_TOKEN,
    POSTGRES_DATABASE,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
    init_environment,
)
from discord import app_commands
from discord.ext import commands
from leaderboard_db import LeaderboardDB
from utils import setup_logging

logger = setup_logging()


class ClusterBot(commands.Bot):
    def __init__(self, debug_mode=False):
        intents = discord.Intents.default()
        intents.members = True
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

        self.leaderboard_db = LeaderboardDB(
            POSTGRES_HOST,
            POSTGRES_DATABASE,
            POSTGRES_USER,
            POSTGRES_PASSWORD,
            POSTGRES_PORT,
        )

    async def setup_hook(self):
        logger.info(f"Syncing commands for staging guild {DISCORD_CLUSTER_STAGING_ID}")
        try:
            # Load cogs
            await self.add_cog(ModalCog(self))
            await self.add_cog(GitHubCog(self))
            await self.add_cog(BotManagerCog(self))
            await self.add_cog(LeaderboardCog(self))
            await self.add_cog(VerifyRunCog(self))

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

    async def _setup_leaderboards(self):  # noqa: C901
        assert len(self.guilds) == 1, "Bot must be in only one guild"

        guild = self.guilds[0]

        category = discord.utils.get(guild.categories, name="Leaderboards")

        if not category:
            category = await guild.create_category(
                name="Leaderboards", reason="Created for leaderboard management"
            )
            logger.info(f"Created new Leaderboards category with ID: {category.id}")

        forum_channel = None
        submission_channel = None
        general_channel = None
        for channel in category.channels:
            if channel.name == "central" and isinstance(channel, discord.ForumChannel):
                forum_channel = channel
            elif channel.name == "submissions" and isinstance(channel, discord.TextChannel):
                submission_channel = channel
            elif channel.name == "general" and isinstance(channel, discord.TextChannel):
                general_channel = channel

        if not forum_channel:
            forum_channel = await category.create_forum(
                name="central", reason="Created for leaderboard management"
            )

        if not general_channel:
            general_channel = await category.create_text_channel(
                name="general", reason="Created for leaderboard general"
            )

        if not submission_channel:
            submission_channel = await category.create_text_channel(
                name="submissions", reason="Created for leaderboard submissions"
            )

        self.leaderboard_forum_id = forum_channel.id
        self.leaderboard_submissions_id = submission_channel.id
        self.leaderboard_general_id = general_channel.id

        leaderboard_admin_role = None
        leaderboard_creator_role = None
        leaderboard_participant_role = None

        for role in category.guild.roles:
            if role.name == "Leaderboard Admin":
                leaderboard_admin_role = role
            elif role.name == "Leaderboard Creator":
                leaderboard_creator_role = role
            elif role.name == "Leaderboard Participant":
                leaderboard_participant_role = role

        if not leaderboard_admin_role:
            leaderboard_admin_role = await category.guild.create_role(
                name="Leaderboard Admin",
                color=discord.Color.purple(),
                reason="Created for leaderboard management",
                permissions=discord.Permissions(
                    manage_channels=True,
                    manage_messages=True,
                    manage_threads=True,
                    view_channel=True,
                    send_messages=True,
                    manage_roles=True,
                ),
            )
            logger.info(
                f"Created leaderboard admin role: {leaderboard_admin_role.name}, please assign this role to the leaderboard admin group in the discord server."  # noqa: E501
            )
        if not leaderboard_creator_role:
            leaderboard_creator_role = await category.guild.create_role(
                name="Leaderboard Creator",
                color=discord.Color.blue(),
                reason="Created for leaderboard management",
            )
            logger.info(
                f"Created leaderboard creator role: {leaderboard_creator_role.name}, please assign this role to the leaderboard creator group in the discord server."  # noqa: E501
            )
        if not leaderboard_participant_role:
            leaderboard_participant_role = await category.guild.create_role(
                name="Leaderboard Participant",
                color=discord.Color.pink(),
                reason="Created for leaderboard management",
            )
            logger.info(
                f"Created leaderboard participant role: {leaderboard_participant_role.name}, please assign this role to the leaderboard participant group in the discord server."  # noqa: E501
            )

        self.leaderboard_admin_role_id = leaderboard_admin_role.id
        self.leaderboard_creator_role_id = leaderboard_creator_role.id
        self.leaderboard_participant_role_id = leaderboard_participant_role.id

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

        await self._setup_leaderboards()

    async def create_thread(
        self, interaction: discord.Interaction, gpu_name: str, job_name: str
    ) -> discord.Thread:
        thread = await interaction.channel.create_thread(
            name=f"{job_name} ({gpu_name}) - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            type=discord.ChannelType.public_thread,
            auto_archive_duration=1440,
        )
        return thread

    async def send_chunked_message(self, channel, content: str, code_block: bool = True):
        """
        Send a long message in chunks to avoid Discord's message length limit

        Args:
            channel: The discord channel/thread to send to
            content: The content to send
            code_block: Whether to wrap the content in code blocks
        """
        chunk_size = 1900  # Leave room for code block syntax
        chunks = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]

        for i, chunk in enumerate(chunks):
            if code_block:
                await channel.send(f"```\nOutput (part {i + 1}/{len(chunks)}):\n{chunk}\n```")
            else:
                await channel.send(chunk)


def main():
    init_environment()

    parser = argparse.ArgumentParser(description="Run the Discord Cluster Bot")
    parser.add_argument("--debug", action="store_true", help="Run in debug/staging mode")
    args = parser.parse_args()

    logger.info("Starting bot...")
    token = DISCORD_DEBUG_TOKEN if args.debug else DISCORD_TOKEN

    if args.debug and not token:
        raise ValueError("DISCORD_DEBUG_TOKEN not found")

    client = ClusterBot(debug_mode=args.debug)
    client.run(token)


if __name__ == "__main__":
    main()
