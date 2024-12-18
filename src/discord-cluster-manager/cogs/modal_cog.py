import time

import discord
import modal
from discord import app_commands
from discord.ext import commands
from utils import send_discord_message, setup_logging

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
    ) -> discord.Thread:
        thread = None
        try:
            if not script.filename.endswith(".py") and not script.filename.endswith(".cu"):
                await send_discord_message(
                    "Please provide a Python (.py) or CUDA (.cu) file"
                )
                return None

            thread = await self.bot.create_thread(interaction, gpu_type.name, "Modal Job")
            queue_start_time = time.perf_counter()
            message = f"Created thread {thread.mention} for your Modal job"

            await send_discord_message(interaction, message)

            await thread.send(f"**Processing `{script.filename}` with {gpu_type.name}...**")

            script_content = (await script.read()).decode("utf-8")
            status_msg = await thread.send("**Running on Modal...**\n> ⏳ Waiting for available GPU...")

            result, execution_time_ms = await self.trigger_modal_run(script_content, script.filename)

            # Update status message to show completion
            await status_msg.edit(content="**Running on Modal...**\n> ✅ Job completed!")

            queue_end_time = time.perf_counter()
            queue_time_ms = (queue_end_time - queue_start_time) * 1000

            # Send metrics and results
            await thread.send(f"\n**Script size:** {len(script_content)} bytes")
            await thread.send(f"**Queue time:** {queue_time_ms:.3f} ms")
            await thread.send(f"**Execution time:** {execution_time_ms:.3f} ms\n")
            await thread.send(f"**Modal execution result:**\n```\n{result}\n```")

            return thread

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            if thread:
                # Update status message to show error
                await status_msg.edit(content="**Running on Modal...**\n> ❌ Job failed!")
                await thread.send(f"**Error:** {str(e)}")
            raise

    async def trigger_modal_run(self, script_content: str, filename: str) -> tuple[str, float]:
        logger.info("Attempting to trigger Modal run")

        from modal_runner import modal_app

        try:
            print(f"Running {filename} with Modal")
            with modal.enable_output():
                with modal_app.run():
                    if filename.endswith(".py"):
                        from modal_runner import run_pytorch_script
                        result, execution_time_ms = run_pytorch_script.remote(script_content)
                    elif filename.endswith(".cu"):
                        from modal_runner import run_cuda_script
                        result, execution_time_ms = run_cuda_script.remote(script_content)

            return result, execution_time_ms

        except Exception as e:
            logger.error(f"Error in trigger_modal_run: {str(e)}", exc_info=True)
            return f"Error: {str(e)}", 0
