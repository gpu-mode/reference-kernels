import json

import discord
from consts import GPUType
from discord import app_commands
from discord.ext import commands
from github_runner import GitHubRun
from leaderboard_eval import amd_requirements, nvidia_requirements
from report import generate_report
from run_eval import CompileResult, FullResult, RunResult
from utils import build_task_config, send_discord_message, setup_logging

logger = setup_logging()


class GitHubCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.run_submission = bot.run_group.command(
            name="github", description="Run a script using GitHub Actions"
        )(self.run_submission)

    @app_commands.describe(
        script="The Python script file to run",
        gpu_type="Choose the GPU type for GitHub Actions",
    )
    @app_commands.choices(
        gpu_type=[
            app_commands.Choice(name="NVIDIA", value="nvidia"),
            app_commands.Choice(name="AMD", value="amd"),
        ]
    )
    async def run_submission(
        self,
        interaction: discord.Interaction,
        script: discord.Attachment,
        gpu_type: app_commands.Choice[str],
        reference_script: discord.Attachment = None,
        reference_code: str = None,
    ) -> tuple[discord.Thread, FullResult]:
        if not script.filename.endswith((".py", ".cu", ".cuh", ".cpp")):
            await send_discord_message(
                interaction, "Please provide a Python (.py) or CUDA (.cu / .cuh / .cpp) file"
            )
            return None, None

        thread = await self.bot.create_thread(interaction, gpu_type.name, "GitHub Job")
        await thread.send(f"Processing `{script.filename}` with {gpu_type.name}...")

        try:
            script_content = (await script.read()).decode("utf-8")
            selected_gpu = GPUType.AMD if gpu_type.value == "amd" else GPUType.NVIDIA
            lang = "py" if script.filename.endswith(".py") else "cu"

            if reference_script is not None or reference_code is not None:
                reference_content = (
                    reference_code
                    if reference_code is not None
                    else (await reference_script.read()).decode("utf-8")
                )
            else:
                reference_content = None

            config = build_task_config(
                lang=lang,
                reference_content=reference_content,
                submission_content=script_content,
                arch=None,
            )

            artifacts = await self.execute_github_run(
                gpu_type=selected_gpu,
                config=config,
                thread=thread,
            )

            logs = artifacts["run-result"]["result.json"].decode("utf-8")
            data = json.loads(logs)
            if "compile" in data and data["compile"] is not None:
                comp = CompileResult(**data["compile"])
            else:
                comp = None
            run = RunResult(**data["run"])
            result = FullResult(success=True, error="", compile=comp, run=run)
            await generate_report(thread, result)
            return thread, result

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            if thread:
                await thread.send(f"Error processing request: {str(e)}")
            raise

    async def execute_github_run(
        self,
        gpu_type: GPUType,
        config: dict,
        thread: discord.Thread,
    ) -> dict:
        lang = config["lang"]
        if lang == "cu" and gpu_type == GPUType.AMD:
            # TODO implement HIP
            raise ValueError("Cannot use CUDA runs with AMD GPUs")

        lang_name = {"py": "Python", "cu": "CUDA"}[lang]

        logger.info(f"Attempting to trigger GitHub action for {lang_name} on {gpu_type.name}")

        workflow_file = gpu_type.value
        run = GitHubRun(workflow_file)

        try:
            payload = json.dumps(config)

            inputs = {"payload": payload}
            if lang == "py":
                if gpu_type == GPUType.NVIDIA:
                    inputs["requirements"] = nvidia_requirements
                else:
                    inputs["requirements"] = amd_requirements

            if not await run.trigger(inputs):
                await thread.send(
                    "Failed to trigger GitHub Action. Please check the configuration."
                )
                return {}

            status_msg = await thread.send(
                "**Running on GitHub...**\n" "> ⏳ Waiting for workflow to start..."
            )
            await run.wait_for_completion(lambda x: self.wait_callback(x, thread, status_msg))
            await thread.send(f"Running completed with status: {run.status}")

            return await run.download_artifacts()

        except Exception as e:
            logger.error(f"Error in trigger_github_action: {str(e)}", exc_info=True)
            raise

    async def wait_callback(self, run: GitHubRun, thread: discord.Thread, msg: discord.Message):
        message = (
            f"**Running on GitHub...**\n"
            f"> Workflow [{run.run_id}]({run.html_url}): {run.status}\n"
            f"> ⏳ {run.elapsed_time.total_seconds():.1f} seconds\n"
        )

        await msg.edit(content=message)
