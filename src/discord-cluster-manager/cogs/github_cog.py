import discord
from discord import app_commands
from discord.ext import commands
from datetime import datetime, timezone
import asyncio
import requests
import zipfile
import os
from github import Github
from utils import setup_logging, get_github_branch_name
from consts import GPUType, GITHUB_TOKEN, GITHUB_REPO

logger = setup_logging()


class GitHubCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.run_github = bot.run_group.command(
            name="github", description="Run a script using GitHub Actions"
        )(self.run_github)

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
    async def run_github(
        self,
        interaction: discord.Interaction,
        script: discord.Attachment,
        gpu_type: app_commands.Choice[str],
    ):
        if not script.filename.endswith(".py") and not script.filename.endswith(".cu"):
            await interaction.response.send_message(
                "Please provide a Python (.py) or CUDA (.cu) file"
            )
            return

        thread = await self.bot.create_thread(interaction, gpu_type.name, "GitHub Job")

        await interaction.response.send_message(
            f"Created thread {thread.mention} for your GitHub job"
        )
        await thread.send(f"Processing `{script.filename}` with {gpu_type.name}...")

        try:
            script_content = (await script.read()).decode("utf-8")
            selected_gpu = GPUType.AMD if gpu_type.value == "amd" else GPUType.NVIDIA
            run_id = await self.trigger_github_action(
                script_content, script.filename, selected_gpu
            )

            if run_id:
                await thread.send(
                    f"GitHub Action triggered! Run ID: {run_id}\nMonitoring progress..."
                )
                status, logs, url = await self.check_workflow_status(run_id, thread)

                await thread.send(f"Training completed with status: {status}")

                if len(logs) > 1900:
                    await self.bot.send_chunked_message(thread, logs, code_block=True)
                else:
                    await thread.send(f"```\nLogs:\n{logs}\n```")

                if url:
                    await thread.send(f"View the full run at: {url}")
            else:
                await thread.send(
                    "Failed to trigger GitHub Action. Please check the configuration."
                )

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            await thread.send(f"Error processing request: {str(e)}")

    async def trigger_github_action(self, script_content, filename, gpu_type):
        logger.info(f"Attempting to trigger GitHub action for {gpu_type.name} GPU")
        gh = Github(GITHUB_TOKEN)
        repo = gh.get_repo(GITHUB_REPO)

        try:
            trigger_time = datetime.now(timezone.utc)
            workflow_file = gpu_type.value
            workflow = repo.get_workflow(workflow_file)

            success = workflow.create_dispatch(
                get_github_branch_name(),
                {"script_content": script_content, "filename": filename},
            )

            if success:
                await asyncio.sleep(2)
                runs = list(workflow.get_runs())

                for run in runs:
                    if run.created_at.replace(tzinfo=timezone.utc) > trigger_time:
                        return run.id
            return None

        except Exception as e:
            logger.error(f"Error in trigger_github_action: {str(e)}", exc_info=True)
            return None

    async def check_workflow_status(self, run_id, thread):
        logger.info(f"Starting to monitor workflow status for run {run_id}")
        gh = Github(GITHUB_TOKEN)
        repo = gh.get_repo(GITHUB_REPO)

        while True:
            try:
                run = repo.get_workflow_run(run_id)

                if run.status == "completed":
                    logs = await self.download_artifact(run_id)
                    return run.conclusion, logs, run.html_url

                await thread.send(
                    f"Workflow still running... Status: {run.status}\nLive view: {run.html_url}"
                )
                await asyncio.sleep(60)
            except Exception as e:
                return "error", str(e), None

    async def download_artifact(self, run_id):
        logger.info(f"Attempting to download artifacts for run {run_id}")
        gh = Github(GITHUB_TOKEN)
        repo = gh.get_repo(GITHUB_REPO)

        try:
            run = repo.get_workflow_run(run_id)
            artifacts = run.get_artifacts()

            for artifact in artifacts:
                if artifact.name == "training-artifacts":
                    url = artifact.archive_download_url
                    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
                    response = requests.get(url, headers=headers)

                    if response.status_code == 200:
                        with open("training.log.zip", "wb") as f:
                            f.write(response.content)

                        with zipfile.ZipFile("training.log.zip") as z:
                            log_file = next(
                                (f for f in z.namelist() if f.endswith("training.log")),
                                None,
                            )
                            if log_file:
                                with z.open(log_file) as f:
                                    logs = f.read().decode("utf-8")
                            else:
                                logs = "training.log file not found in artifact"

                        os.remove("training.log.zip")
                        return logs
                    else:
                        return f"Failed to download artifact. Status code: {response.status_code}"

            return "No training artifacts found"
        except Exception as e:
            return f"Error downloading artifacts: {str(e)}"
