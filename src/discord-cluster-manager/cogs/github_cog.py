import asyncio
import json
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Optional

import discord
import requests
from consts import GPUType
from discord import app_commands
from discord.ext import commands
from env import GITHUB_REPO, GITHUB_TOKEN
from github import Github
from leaderboard_eval import amd_requirements, nvidia_requirements
from report import generate_report
from run_eval import CompileResult, FullResult, RunResult
from utils import build_task_config, get_github_branch_name, send_discord_message, setup_logging

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

            run_id = await self.trigger_github_run(
                lang=lang,
                gpu_type=selected_gpu,
                script_content=script_content,
                reference_content=reference_content,
            )

            if run_id:
                await thread.send(
                    f"GitHub Action triggered! Run ID: {run_id}\nMonitoring progress..."
                )
                status, result, url = await self.check_workflow_status(run_id, thread, gpu_type)

                await thread.send(f"Training completed with status: {status}")

                if isinstance(result, FullResult):
                    await generate_report(thread, result)
                else:
                    if len(result) > 1900:
                        await self.bot.send_chunked_message(thread, result, code_block=True)
                    else:
                        await thread.send(f"```\nLogs:\n{result}\n```")

                if url:
                    await thread.send(f"View the full run at: <{url}>")
            else:
                await thread.send(
                    "Failed to trigger GitHub Action. Please check the configuration."
                )

            return thread, result

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            if thread:
                await thread.send(f"Error processing request: {str(e)}")
            raise

    async def trigger_github_run(
        self, lang: str, gpu_type: GPUType, script_content: str, reference_content: Optional[str]
    ):
        if lang == "cu" and gpu_type == GPUType.AMD:
            # TODO implement HIP
            raise ValueError("Cannot use CUDA runs with AMD GPUs")

        lang_name = {"py": "Python", "cu": "CUDA"}[lang]

        config = build_task_config(
            lang=lang,
            reference_content=reference_content,
            submission_content=script_content,
            arch=None,
        )

        logger.info(f"Attempting to trigger GitHub action for {lang_name} on {gpu_type.name}")
        gh = Github(GITHUB_TOKEN)
        repo = gh.get_repo(GITHUB_REPO)

        try:
            trigger_time = datetime.now(timezone.utc)
            workflow_file = gpu_type.value
            workflow = repo.get_workflow(workflow_file)

            payload = json.dumps(config)

            inputs = {"payload": payload}
            if lang == "py":
                if gpu_type == GPUType.NVIDIA:
                    inputs["requirements"] = nvidia_requirements
                else:
                    inputs["requirements"] = amd_requirements

            success = workflow.create_dispatch(get_github_branch_name(), inputs=inputs)
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

    async def check_workflow_status(self, run_id, thread, gpu_type):
        logger.info(f"Starting to monitor workflow status for run {run_id}")
        gh = Github(GITHUB_TOKEN)
        repo = gh.get_repo(GITHUB_REPO)
        start_time = datetime.now(timezone.utc)
        timeout_minutes = 5
        timeout = timedelta(minutes=timeout_minutes)

        while True:
            try:
                run = repo.get_workflow_run(run_id)
                elapsed_time = datetime.now(timezone.utc) - start_time

                if elapsed_time > timeout:
                    try:
                        run.cancel()
                        # Wait briefly to ensure cancellation is processed
                        # And Verify the run was actually cancelled
                        await asyncio.sleep(5)
                        run = repo.get_workflow_run(run_id)
                        if run.status != "completed":
                            logger.warning(f"Failed to cancel workflow run {run_id}")
                    except Exception as e:
                        logger.error(f"Error cancelling workflow: {str(e)}")

                    await thread.send(
                        f"Workflow cancelled - exceeded {timeout_minutes} minute timeout"
                    )
                    return (
                        "cancelled",
                        f"Workflow exceeded {timeout_minutes} minute timeout",
                        run.html_url,
                    )

                if run.status == "completed":
                    result = await self.download_results(run_id)
                    return run.conclusion, result, run.html_url

                await thread.send(
                    f"Workflow: {run.status} running for "
                    f"{elapsed_time.total_seconds():.2f} seconds\n"
                    f"Live view: <{run.html_url}>"
                )
                await asyncio.sleep(20)
            except Exception as e:
                logger.error("Error", exc_info=e)
                return "error", str(e), None

    async def download_results(self, run_id) -> FullResult:
        try:
            data = await self.download_artifact(run_id, name="run-result")
            logs = data["result.json"].decode("utf-8")
            data = json.loads(logs)
            if "compile" in data and data["compile"] is not None:
                comp = CompileResult(**data["compile"])
            else:
                comp = None
            run = RunResult(**data["run"])
            return FullResult(success=True, error="", compile=comp, run=run)
        except Exception as e:
            logger.error("Error downloading artifacts", exc_info=e)
            return FullResult(
                success=False,
                error=f"Error downloading artifacts: {repr(e)}",
                compile=None,
                run=None,
            )

    async def download_artifact(self, run_id, name: str):
        logger.info(f"Attempting to download artifact {name} for run {run_id}")
        gh = Github(GITHUB_TOKEN)
        repo = gh.get_repo(GITHUB_REPO)

        run = repo.get_workflow_run(run_id)
        artifacts = run.get_artifacts()

        for artifact in artifacts:
            if artifact.name == name:
                url = artifact.archive_download_url
                headers = {"Authorization": f"token {GITHUB_TOKEN}"}
                response = requests.get(url, headers=headers)

                if response.status_code == 200:
                    with tempfile.NamedTemporaryFile("w+b") as temp:
                        temp.write(response.content)
                        temp.flush()

                        with zipfile.ZipFile(temp.name) as z:
                            artifact_dict = {}
                            for file in z.namelist():
                                with z.open(file) as f:
                                    artifact_dict[file] = f.read()

                    return artifact_dict
                else:
                    raise RuntimeError(
                        f"Failed to download artifact. Status code: {response.status_code}"
                    )
        raise RuntimeError(f"Could not find artifact {name}")
