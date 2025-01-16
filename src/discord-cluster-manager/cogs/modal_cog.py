import asyncio
from typing import Optional

import discord
import modal
from consts import GPU_TO_SM, ModalGPU
from discord import app_commands
from discord.ext import commands
from report import generate_report
from run_eval import FullResult
from utils import build_task_config, send_discord_message, setup_logging

logger = setup_logging()


class ModalCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

        self.run_submission = bot.run_group.command(
            name="modal", description="Run a script using Modal"
        )(self.run_submission)

    @app_commands.describe(
        script="The Python script file to run", gpu_type="Choose the GPU type for Modal"
    )
    @app_commands.choices(
        gpu_type=[app_commands.Choice(name=gpu.name, value=gpu.value) for gpu in ModalGPU]
    )
    async def run_submission(
        self,
        interaction: discord.Interaction,
        script: discord.Attachment,
        gpu_type: app_commands.Choice[str],
        reference_script: Optional[discord.Attachment] = None,
        reference_code: str = None,
    ) -> tuple[discord.Thread, FullResult]:
        thread = None
        status_msg = None
        try:
            if not script.filename.endswith((".py", ".cu", ".cuh", ".cpp")):
                await send_discord_message(
                    interaction,
                    "Please provide a Python (.py) or CUDA (.cu / .cuh / .cpp) file",
                    ephemeral=True,
                )
                return None, None

            if not interaction.response.is_done():
                await interaction.response.defer(ephemeral=True)

            channel = interaction.channel
            message = await channel.send(f"Starting Modal job with {gpu_type.name}...")
            thread = await message.create_thread(name=f"{gpu_type.name} Modal Job")

            script_content = (await script.read()).decode("utf-8")
            status_msg = await thread.send(
                "**Running on Modal...**\n> ⏳ Waiting for available GPU..."
            )

            filename = "submission.py" if script.filename.endswith(".py") else "train.cu"
            reference_content = None
            if reference_script is not None or reference_code is not None:
                reference_content = (
                    reference_code
                    if reference_code is not None
                    else (await reference_script.read()).decode("utf-8")
                )

            config = build_task_config(
                lang="py" if filename.endswith(".py") else "cu",
                reference_content=reference_content,
                submission_content=script_content,
                arch=GPU_TO_SM[gpu_type.value.upper()],
            )

            result = await self.handle_modal_execution(
                thread,
                gpu_type.value,
                config,
                status_msg,
            )
            return thread, result

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            if thread and status_msg:
                await status_msg.edit(content="**Running on Modal...**\n> ❌ Job failed!")
                await thread.send(f"**Error:** {str(e)}")
            raise

    async def handle_modal_execution(
        self,
        thread: discord.Thread,
        gpu_type: str,
        config: dict,
        status_msg: discord.Message,
    ) -> FullResult:
        try:
            loop = asyncio.get_event_loop()
            func_type = "pytorch" if config["lang"] == "py" else "cuda"
            func_name = f"run_{func_type}_script_{gpu_type.lower()}"

            result = await loop.run_in_executor(
                None,
                lambda: modal.Function.lookup("discord-bot-runner", func_name).remote(
                    config=config
                ),
            )

            # Send results
            await generate_report(thread, result)
            return result

        except Exception as e:
            logger.error(f"Error in handle_modal_execution: {str(e)}", exc_info=True)
            await status_msg.edit(content="**Running on Modal...**\n> ❌ Job failed!")
            await thread.send(f"**Error:** {str(e)}")
            raise
