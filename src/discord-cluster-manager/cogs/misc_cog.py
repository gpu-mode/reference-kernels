import discord
from discord import app_commands
from discord.ext import commands
from utils import setup_logging

logger = setup_logging()


class BotManagerCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @app_commands.command(name="ping")
    async def ping(self, interaction: discord.Interaction):
        """Simple ping command to check if the bot is responsive"""
        await interaction.response.send_message("pong")

    @app_commands.command(name="resync")
    async def resync(self, interaction: discord.Interaction):
        """Admin command to resync slash commands"""
        if interaction.user.guild_permissions.administrator:
            try:
                await interaction.response.defer()
                # Clear and resync
                self.bot.tree.clear_commands(guild=interaction.guild)
                await self.bot.tree.sync(guild=interaction.guild)
                commands = await self.bot.tree.fetch_commands(guild=interaction.guild)
                await interaction.followup.send(
                    "Resynced commands:\n"
                    + "\n".join([f"- /{cmd.name}" for cmd in commands])
                )
            except Exception as e:
                logger.error(f"Error in resync command: {str(e)}", exc_info=True)
                await interaction.followup.send(f"Error: {str(e)}")
        else:
            await interaction.response.send_message(
                "You need administrator permissions to use this command"
            )
