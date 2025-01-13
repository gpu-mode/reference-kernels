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
from leaderboard_eval import cu_eval, py_eval
from report import generate_report
from run_eval import CompileResult, FullResult, RunResult
from utils import get_github_branch_name, send_discord_message, setup_logging

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
    ) -> discord.Thread:
        if not script.filename.endswith((".py", ".cu", ".cuh", ".cpp")):
            await send_discord_message(
                interaction, "Please provide a Python (.py) or CUDA (.cu / .cuh / .cpp) file"
            )
            return None

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

            if gpu_type.value == "nvidia":
                run_id = await self.trigger_github_nvidia(
                    lang=lang,
                    script_content=script_content,
                    reference_content=reference_content,
                )
            else:
                ##########
                # OLD CODE
                filename = "train.py" if script.filename.endswith(".py") else "train.cu"
                if reference_script is not None or reference_code is not None:
                    reference_content = (
                        reference_code
                        if reference_code is not None
                        else (await reference_script.read()).decode("utf-8")
                    )
                    eval_code = py_eval if script.filename.endswith(".py") else cu_eval

                    run_id = await self.trigger_github_amd(
                        script_content,
                        filename,
                        selected_gpu,
                        reference_content,
                        eval_code,
                    )
                else:
                    run_id = await self.trigger_github_amd(script_content, filename, selected_gpu)
                ##########

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

            return thread

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            if thread:
                await thread.send(f"Error processing request: {str(e)}")
            raise

    async def trigger_github_nvidia(
        self, lang: str, script_content: str, reference_content: Optional[str]
    ):
        eval_name = {"py": "eval.py", "cu": "eval.cu"}[lang]
        ref_name = {"py": "reference.py", "cu": "reference.cuh"}[lang]
        sub_name = {"py": "submission.py", "cu": "submission.cuh"}[lang]

        if reference_content is None:
            config = {eval_name: script_content, "lang": lang}
        else:
            config = {ref_name: reference_content, sub_name: script_content, "lang": lang}

        logger.info("Attempting to trigger GitHub action for NVIDIA")
        gh = Github(GITHUB_TOKEN)
        repo = gh.get_repo(GITHUB_REPO)

        try:
            trigger_time = datetime.now(timezone.utc)
            workflow_file = "nvidia_workflow.yml"
            workflow = repo.get_workflow(workflow_file)

            payload = json.dumps(config)

            success = workflow.create_dispatch(
                get_github_branch_name(),
                {
                    "payload": payload,
                },
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

    async def trigger_github_amd(
        self,
        script_content,
        filename,
        gpu_type,
        reference_content=None,
        eval_content=None,
    ):
        logger.info(f"Attempting to trigger GitHub action for {gpu_type.name} GPU")
        gh = Github(GITHUB_TOKEN)
        repo = gh.get_repo(GITHUB_REPO)

        try:
            trigger_time = datetime.now(timezone.utc)
            workflow_file = gpu_type.value
            workflow = repo.get_workflow(workflow_file)

            if reference_content is not None:
                eval_filename = "eval.py" if filename.endswith(".py") else "eval.cu"
                reference_filename = "reference.py" if filename.endswith(".py") else "reference.cuh"
                filename = "train.py" if filename.endswith(".py") else "train.cuh"
                success = workflow.create_dispatch(
                    get_github_branch_name(),
                    {
                        "script_content": script_content,
                        "filename": filename,
                        "reference_content": reference_content,
                        "reference_filename": reference_filename,
                        "eval_content": eval_content,
                        "eval_filename": eval_filename,
                    },
                )
            else:
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
                    if gpu_type.value == "nvidia":
                        result = await self.download_results(run_id)
                    else:
                        result = await self.handle_training_log(run_id)
                    return run.conclusion, result, run.html_url

                await thread.send(
                    f"Workflow: {run.status} running for "
                    f"{elapsed_time.total_seconds():.2f} seconds\n"
                    f"Live view: <{run.html_url}>"
                )
                await asyncio.sleep(20)
            except Exception as e:
                return "error", str(e), None

    async def download_results(self, run_id) -> FullResult:
        try:
            data = await self.download_artifact(run_id, name="run-result")
            logs = data["result.json"].decode("utf-8")
            data = json.loads(logs)
            if "compile" in data:
                comp = CompileResult(**data["compile"])
            else:
                comp = None
            run = RunResult(**data["run"])
            return FullResult(success=True, error="", compile=comp, run=run)
        except Exception as e:
            return FullResult(
                success=False,
                error=f"Error downloading artifacts: {str(e)}",
                compile=None,
                run=None,
            )

    async def handle_training_log(self, run_id):
        try:
            data = await self.download_artifact(run_id, name="training-artifacts")
            logs = data["training.log"].decode("utf-8")
            return logs
        except Exception as e:
            return f"Error downloading artifacts: {str(e)}"

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
        return RuntimeError(f"Could not find artifact {name}")
