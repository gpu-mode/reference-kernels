import discord
from discord import ui
from datetime import datetime

from discord.ext import commands


def generate_modal():
    class LeaderboardModal(ui.Modal, title="Create Leaderboard"):
        name = ui.TextInput(
            label="Leaderboard Name",
            placeholder="Enter the leaderboard name",
            required=True,
            max_length=100
        )
        
        date = ui.TextInput(
            label="Date",
            placeholder="YYYY-MM-DD HH:MM",
            required=True
        )
        
        template_code = ui.TextInput(
            label="Template Code",
            style=discord.TextStyle.paragraph,
            placeholder="Paste your template code here",
            required=True
        )
        
        async def on_submit(self, interaction: discord.Interaction):
            try:
                # Parse the date string to datetime
                date_value = datetime.strptime(str(self.date), "%Y-%m-%d %H:%M")
                await interaction.response.send_message(
                    f"Leaderboard '{self.name}' created for {date_value}",
                    ephemeral=True
                )
            except ValueError:
                await interaction.response.send_message(
                    "Invalid date format. Please use YYYY-MM-DD HH:MM",
                    ephemeral=True
                )
    
    return LeaderboardModal()


class LeaderboardCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.get_leaderboard = bot.leaderboard_group.command(name="get")(
            self.get_leaderboard
        )
        self.leaderboard_create = bot.leaderboard_group.command(name="create")(
            self.leaderboard_create
        )

    async def get_leaderboard(self, interaction: discord.Interaction):
        await interaction.response.send_message("Leaderboard")
    

    async def leaderboard_create(self, interaction: discord.Interaction):
        modal = generate_modal()
        await interaction.response.send_modal(modal)

