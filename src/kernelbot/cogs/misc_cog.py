import os
from typing import TYPE_CHECKING

import discord
import psycopg2
from discord import app_commands
from discord.ext import commands

from kernelbot.discord_utils import send_discord_message
from kernelbot.env import env
from libkernelbot.utils import setup_logging

if TYPE_CHECKING:
    from ..main import ClusterBot

logger = setup_logging()


class BotManagerCog(commands.Cog):
    def __init__(self, bot: "ClusterBot"):
        self.bot = bot

    @app_commands.command(name="ping")
    async def ping(self, interaction: discord.Interaction):
        """Simple ping command to check if the bot is responsive"""
        await send_discord_message(interaction, "pong")

    @app_commands.command(name="verifydb")
    async def verify_db(self, interaction: discord.Interaction):
        """Command to verify database connectivity"""
        if not env.DATABASE_URL:
            message = "DATABASE_URL not set."
            logger.error(message)
            await send_discord_message(interaction, message)
            return

        try:
            sslmode = "disable" if env.DISABLE_SSL else "require"
            with psycopg2.connect(env.DATABASE_URL, sslmode=sslmode) as conn:
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

    @app_commands.command(name="get-api-url")
    async def get_api_url(self, interaction: discord.Interaction):
        if not os.environ.get("HEROKU_APP_DEFAULT_DOMAIN_NAME"):
            await send_discord_message(
                interaction,
                "No `HEROKU_APP_DEFAULT_DOMAIN_NAME` present,"
                " are you sure you aren't running locally?",
                ephemeral=True,
            )
        else:
            await send_discord_message(
                interaction,
                f"API URL: `https://{os.environ['HEROKU_APP_DEFAULT_DOMAIN_NAME'].rstrip('/')}`",
                ephemeral=True,
            )
