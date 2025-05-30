import asyncio
import datetime
import json
import math
import pprint
import tempfile
import zipfile
from typing import Awaitable, Callable, Optional

import requests
from consts import (
    AMD_REQUIREMENTS,
    DEFAULT_GITHUB_TIMEOUT_MINUTES,
    GPU,
    NVIDIA_REQUIREMENTS,
    TIMEOUT_BUFFER_MINUTES,
    GitHubGPU,
    SubmissionMode,
)
from github import Github, UnknownObjectException, WorkflowRun
from report import RunProgressReporter
from run_eval import CompileResult, EvalResult, FullResult, RunResult, SystemInfo
from utils import get_github_branch_name, setup_logging

from .launcher import Launcher

logger = setup_logging()

def get_timeout(config: dict) -> int:
    mode = config.get("mode")
    sec_map = {
        SubmissionMode.TEST.value: config.get("test_timeout"),
        SubmissionMode.BENCHMARK.value: config.get("benchmark_timeout"),
        SubmissionMode.LEADERBOARD.value: config.get("ranked_timeout"),
    }
    seconds = sec_map.get(mode) or DEFAULT_GITHUB_TIMEOUT_MINUTES * 60
    return math.ceil(seconds / 60)

class GitHubLauncher(Launcher):
    def __init__(self, repo: str, token: str):
        super().__init__(name="GitHub", gpus=GitHubGPU)
        self.repo = repo
        self.token = token
        self.trigger_limit = asyncio.Semaphore(1)

    async def run_submission(
        self, config: dict, gpu_type: GPU, status: RunProgressReporter
    ) -> FullResult:
        gpu_vendor = None
        if gpu_type.value in ["MI300", "MI250"]:
            selected_workflow = "amd_workflow.yml"
            runner_name = {"MI300": "amdgpu-mi300-x86-64", "MI250": "amdgpu-mi250-x86-64"}[
                gpu_type.value
            ]
            gpu_vendor = "AMD"
            requirements = AMD_REQUIREMENTS
        elif gpu_type.value == "NVIDIA":
            selected_workflow = "nvidia_workflow.yml"
            gpu_vendor = "NVIDIA"
            requirements = NVIDIA_REQUIREMENTS
        else:
            raise ValueError(f"Invalid GPU type: {gpu_type.value}")

        lang = config["lang"]
        if lang == "cu" and gpu_vendor == "AMD":
            # TODO implement HIP
            raise NotImplementedError("Cannot use CUDA runs with AMD GPUs")

        lang_name = {"py": "Python", "cu": "CUDA"}[lang]

        logger.info(f"Attempting to trigger GitHub action for {lang_name} on {selected_workflow}")
        run = GitHubRun(self.repo, self.token, selected_workflow)
        logger.info(f"Successfully created GitHub run: {run.run_id}")

        payload = json.dumps(config)

        inputs = {"payload": payload}
        if lang == "py":
            inputs["requirements"] = requirements
            if gpu_vendor == "AMD":
                inputs["runner"] = runner_name

        async with self.trigger_limit:  # DO NOT REMOVE, PREVENTS A RACE CONDITION
            if not await run.trigger(inputs):
                raise RuntimeError(
                    "Failed to trigger GitHub Action. Please check the configuration."
                )

        await status.push("⏳ Waiting for workflow to start...")
        logger.info("Waiting for workflow to start...")

        timeout = get_timeout(config) + TIMEOUT_BUFFER_MINUTES
        logger.info(f"Waiting for workflow to complete... (timeout: {timeout} minutes)")
        await run.wait_for_completion(
            lambda x: self.wait_callback(x, status),
            timeout_minutes=timeout
            )
        await status.update(f"Workflow [{run.run_id}]({run.html_url}) completed")
        logger.info(f"Workflow [{run.run_id}]({run.html_url}) completed")
        await status.push("Downloading artifacts...")
        logger.info("Downloading artifacts...")

        artifacts = await run.download_artifacts()
        if "run-result" not in artifacts:
            logger.error("Could not find `run-result` among artifacts: %s", artifacts.keys())
            await status.push("Downloading artifacts...  failed")
            return FullResult(
                success=False, error="Could not download artifacts", runs={}, system=SystemInfo()
            )

        logs = artifacts["run-result"]["result.json"].decode("utf-8")

        await status.update("Downloading artifacts... done")
        logger.info("Downloading artifacts... done")

        data = json.loads(logs)
        runs = {}
        # convert json back to EvalResult structures, which requires
        # special handling for datetime and our dataclasses.
        for k, v in data["runs"].items():
            if "compilation" in v and v["compilation"] is not None:
                comp = CompileResult(**v["compilation"])
            else:
                comp = None
            run = RunResult(**v["run"])
            res = EvalResult(
                start=datetime.datetime.fromisoformat(v["start"]),
                end=datetime.datetime.fromisoformat(v["end"]),
                compilation=comp,
                run=run,
            )
            runs[k] = res

        system = SystemInfo(**data.get("system", {}))
        return FullResult(success=True, error="", runs=runs, system=system)

    async def wait_callback(self, run: "GitHubRun", status: RunProgressReporter):
        await status.update(
            f"⏳ Workflow [{run.run_id}]({run.html_url}): {run.status} "
            f"({run.elapsed_time.total_seconds():.1f}s)"
        )


class GitHubRun:
    def __init__(self, repo: str, token: str, workflow_file: str):
        gh = Github(token)
        self.repo = gh.get_repo(repo)
        self.token = token
        self.workflow_file = workflow_file
        self.run: Optional[WorkflowRun.WorkflowRun] = None
        self.start_time = None

    @property
    def run_id(self):
        if self.run is None:
            return None
        return self.run.id

    @property
    def html_url(self):
        if self.run is None:
            return None
        return self.run.html_url

    @property
    def status(self):
        if self.run is None:
            return None
        return self.run.status

    @property
    def elapsed_time(self):
        if self.start_time is None:
            return None
        return datetime.datetime.now(datetime.timezone.utc) - self.start_time

    async def trigger(self, inputs: dict) -> bool:
        """
        Trigger this run with the provided inputs.
        Sets `self.run` to the new WorkflowRun on success.

        Returns: Whether the run was successfully triggered,
        """
        trigger_time = datetime.datetime.now(datetime.timezone.utc)
        try:
            workflow = await asyncio.to_thread(self.repo.get_workflow, self.workflow_file)
        except UnknownObjectException as e:
            logger.error(f"Could not find workflow {self.workflow_file}", exc_info=e)
            raise ValueError(f"Could not find workflow {self.workflow_file}") from e

        branch_name = get_github_branch_name()
        logger.debug(
            "Dispatching workflow %s on branch %s with inputs %s",
            self.workflow_file,
            branch_name,
            pprint.pformat(inputs),
        )
        success = await asyncio.to_thread(workflow.create_dispatch, branch_name, inputs=inputs)

        if success:
            wait_seconds = 5
            logger.info(
                f"Workflow dispatch successful. Waiting {wait_seconds}s for the run to appear..."
            )
            await asyncio.sleep(wait_seconds)
            recent_runs_paginated = await asyncio.to_thread(
                workflow.get_runs, event="workflow_dispatch"
            )

            logger.info(
                f"Checking recent workflow_dispatch runs after {trigger_time.isoformat()}..."
            )
            found_run = None
            runs_checked = 0
            try:
                run_iterator = recent_runs_paginated.__iter__()
                while runs_checked < 50:
                    try:
                        run = next(run_iterator)
                        runs_checked += 1
                        logger.debug(
                            f"Checking run {run.id} created at {run.created_at.isoformat()}"
                        )
                        if run.created_at.replace(
                            tzinfo=datetime.timezone.utc
                        ) > trigger_time - datetime.timedelta(seconds=2):
                            found_run = run
                            logger.info(f"Found matching workflow run: ID {found_run.id}")
                            break
                        else:
                            logger.info(f"Run {run.id} is older than trigger time, stopping check.")
                            break
                    except StopIteration:
                        logger.debug("Reached end of recent runs list.")
                        break
            except Exception as e:
                logger.error(f"Error iterating through recent runs: {e}", exc_info=True)
                return False

            if found_run:
                self.run = found_run
                return True
            else:
                logger.warning(
                    f"Could not find a workflow run created after {trigger_time.isoformat()}."
                )
                return False
        else:
            logger.error(
                f"Failed to dispatch workflow {self.workflow_file} on branch {branch_name}."
            )
            return False

    async def wait_for_completion(
        self, callback: Callable[["GitHubRun"], Awaitable[None]], timeout_minutes: int = 10
    ):
        if self.run is None:
            raise ValueError("Run needs to be triggered before a status check!")

        self.start_time = datetime.datetime.now(datetime.timezone.utc)
        timeout = datetime.timedelta(minutes=timeout_minutes)

        while True:
            try:
                run_update = await asyncio.to_thread(self.repo.get_workflow_run, self.run_id)
                self.run = run = run_update

                if self.elapsed_time > timeout:
                    try:
                        self.run.cancel()
                        # Wait briefly to ensure cancellation is processed
                        # And Verify the run was actually cancelled
                        await asyncio.sleep(5)
                        run = self.repo.get_workflow_run(self.run_id)
                        if run.status != "completed":
                            logger.warning(f"Failed to cancel workflow run {self.run_id}")
                    except Exception as e:
                        logger.error(f"Error cancelling workflow: {str(e)}", exc_info=e)
                        raise

                    logger.warning(
                        f"Workflow {self.run_id} cancelled - "
                        f"exceeded {timeout_minutes} minute timeout"
                    )
                    raise TimeoutError(
                        f"Workflow {self.run_id} cancelled - "
                        f"exceeded {timeout_minutes} minute timeout"
                    )

                if run.status == "completed":
                    return

                await callback(self)
                await asyncio.sleep(20)  # Yield control while waiting
            except TimeoutError:
                raise  # Re-raise the specific TimeoutError from the timeout block
            except Exception as e:
                logger.error(f"Error waiting for GitHub run {self.run_id}: {e}", exc_info=e)
                raise  # Re-raise other exceptions

    async def download_artifacts(self) -> dict:
        logger.info("Attempting to download artifacts for run %s", self.run_id)
        artifacts = self.run.get_artifacts()

        extracted = {}

        for artifact in artifacts:
            url = artifact.archive_download_url
            headers = {"Authorization": f"token {self.token}"}
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

                extracted[artifact.name] = artifact_dict
            else:
                raise RuntimeError(
                    f"Failed to download artifact {artifact.name}. "
                    f"Status code: {response.status_code}"
                )

        logger.info("Download artifacts for run %s: %s", self.run_id, list(extracted.keys()))
        return extracted
