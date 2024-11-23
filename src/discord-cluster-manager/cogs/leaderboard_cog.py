import discord
from datetime import datetime

from discord.ext import commands

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot import ClusterBot


class LeaderboardCog(commands.Cog):
    def __init__(self, bot):
        self.bot: ClusterBot = bot
        self.get_leaderboards = bot.leaderboard_group.command(name="get")(
            self.get_leaderboards
        )
        self.leaderboard_create = bot.leaderboard_group.command(
            name="create", description="Create a new leaderboard"
        )(self.leaderboard_create)

    async def get_leaderboards(self, interaction: discord.Interaction):
        """Display all leaderboards in a table format"""
        await interaction.response.defer()

        with self.bot.leaderboard_db as db:
            leaderboards = db.get_leaderboards()

        if not leaderboards:
            await interaction.followup.send("No leaderboards found.", ephemeral=True)
            return

        # Create embed
        embed = discord.Embed(title="Active Leaderboards", color=discord.Color.blue())

        # Add fields for each leaderboard
        for lb in leaderboards:
            deadline_str = lb["deadline"].strftime("%Y-%m-%d %H:%M")
            embed.add_field(
                name=lb["name"], value=f"Deadline: {deadline_str}", inline=False
            )

        await interaction.followup.send(embed=embed)

    @discord.app_commands.describe(
        name="Name of the leaderboard",
        date="Date in YYYY-MM-DD format (time HH:MM is optional)",
        template_file="Template file to upload",
    )
    async def leaderboard_create(
        self,
        interaction: discord.Interaction,
        name: str,
        date: str,
        template_file: discord.Attachment,
    ):
        try:
            # Try parsing with time first
            try:
                date_value = datetime.strptime(date, "%Y-%m-%d %H:%M")
            except ValueError:
                # If that fails, try parsing just the date (will set time to 00:00)
                date_value = datetime.strptime(date, "%Y-%m-%d")

            # Read the template file
            template_content = await template_file.read()

            with self.bot.leaderboard_db as db:
                db.create_leaderboard(
                    {
                        "name": name,
                        "deadline": date_value,
                        "template_code": template_content.decode("utf-8"),
                    }
                )

            await interaction.response.send_message(
                f"Leaderboard '{name}'. Submission deadline: {date_value}",
                ephemeral=True,
            )
        except ValueError:
            await interaction.response.send_message(
                "Invalid date format. Please use YYYY-MM-DD or YYYY-MM-DD HH:MM",
                ephemeral=True,
            )
