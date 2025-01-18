import discord
import psycopg2
from discord import app_commands
from discord.ext import commands
from env import DATABASE_URL
from utils import send_discord_message, setup_logging

logger = setup_logging()


class BotManagerCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @app_commands.command(name="ping")
    async def ping(self, interaction: discord.Interaction):
        """Simple ping command to check if the bot is responsive"""
        await send_discord_message(interaction, "pong")

    @app_commands.command(name="resync")
    async def resync(self, interaction: discord.Interaction):
        logger.info("Resyncing commands")

        """Admin command to resync slash commands"""
        if interaction.user.guild_permissions.administrator:
            try:
                await interaction.response.defer()
                # Clear and resync
                self.bot.tree.clear_commands(guild=interaction.guild)
                await self.bot.tree.sync(guild=interaction.guild)
                commands = await self.bot.tree.fetch_commands(guild=interaction.guild)
                await send_discord_message(
                    interaction,
                    "Resynced commands:\n" + "\n".join([f"- /{cmd.name}" for cmd in commands]),
                )
            except Exception as e:
                logger.error(f"Error in resync command: {str(e)}", exc_info=True)
                await send_discord_message(interaction, f"Error: {str(e)}")
        else:
            await send_discord_message(
                interaction, "You need administrator permissions to use this command"
            )

    @app_commands.command(name="verifydb")
    async def verify_db(self, interaction: discord.Interaction):
        """Command to verify database connectivity"""
        if not DATABASE_URL:
            message = "DATABASE_URL not set."
            logger.error(message)
            await send_discord_message(interaction, message)
            return

        try:
            with psycopg2.connect(DATABASE_URL, sslmode="require") as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT RANDOM()")
                    result = cursor.fetchone()
                    if result:
                        random_value = result[0]
                        await send_discord_message(
                            interaction, f"Your lucky number is {random_value}."
                        )
                    else:
                        await send_discord_message(interaction, "No result returned.")
        except Exception as e:
            message = "Error interacting with the database"
            logger.error(f"{message}: {str(e)}", exc_info=True)
            await send_discord_message(interaction, f"{message}.")
