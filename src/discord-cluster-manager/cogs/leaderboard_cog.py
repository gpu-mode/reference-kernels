import discord

from discord.ext import commands


class LeaderboardCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.get_leaderboard = bot.leaderboard_group.command(name="get")(
            self.get_leaderboard
        )

    async def get_leaderboard(self, interaction: discord.Interaction):
        await interaction.response.send_message("Leaderboard")
