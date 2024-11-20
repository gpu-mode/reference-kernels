import discord
from discord import app_commands
from discord.ext import commands
import modal
from utils import setup_logging

logger = setup_logging()

class ModalCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

        self.run_modal = bot.run_group.command(
            name="modal",
            description="Run a script using Modal"
        )(self.run_modal)


    @app_commands.describe(
        script="The Python script file to run", gpu_type="Choose the GPU type for Modal"
    )
    @app_commands.choices(
        gpu_type=[
            app_commands.Choice(name="NVIDIA T4", value="t4"),
        ]
    )
    async def run_modal(
        self,
        interaction: discord.Interaction,
        script: discord.Attachment,
        gpu_type: app_commands.Choice[str],
    ):
        if not script.filename.endswith(".py"):
            await interaction.response.send_message(
                "Please provide a Python (.py) file"
            )
            return

        thread = await self.bot.create_thread(interaction, gpu_type.name, "Modal Job")

        await interaction.response.send_message(
            f"Created thread {thread.mention} for your Modal job"
        )
        await thread.send(f"Processing `{script.filename}` with {gpu_type.name}...")

        try:
            script_content = (await script.read()).decode("utf-8")
            await thread.send("Running on Modal...")
            result = await self.trigger_modal_run(script_content, script.filename)
            await thread.send(f"```\nModal execution result:\n{result}\n```")
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            await thread.send(f"Error processing request: {str(e)}")

    async def trigger_modal_run(self, script_content: str, filename: str) -> str:
        logger.info("Attempting to trigger Modal run")

        from modal_runner import modal_app, run_script

        try:
            with modal.enable_output():
                with modal_app.run():
                    result = run_script.remote(script_content)
                return result
        except Exception as e:
            logger.error(f"Error in trigger_modal_run: {str(e)}", exc_info=True)
            return f"Error: {str(e)}"

