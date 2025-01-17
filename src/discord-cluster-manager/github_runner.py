import asyncio
import pprint
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Optional

import requests
from env import GITHUB_REPO, GITHUB_TOKEN
from github import Github, UnknownObjectException, WorkflowRun
from utils import get_github_branch_name, setup_logging

logger = setup_logging()


class GitHubRun:
    def __init__(self, workflow_file: str):
        gh = Github(GITHUB_TOKEN)
        self.repo = gh.get_repo(GITHUB_REPO)
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
        return datetime.now(timezone.utc) - self.start_time

    async def trigger(self, inputs: dict) -> bool:
        """
        Trigger this run with the provided inputs.
        Sets `self.run` to the new WorkflowRun on success.

        Returns: Whether the run was successfully triggered,
        """
        trigger_time = datetime.now(timezone.utc)
        try:
            workflow = self.repo.get_workflow(self.workflow_file)
        except UnknownObjectException as e:
            logger.error(f"Could not find workflow {self.workflow_file}", exc_info=e)
            raise ValueError(f"Could not find workflow {self.workflow_file}") from e

        logger.debug(
            "Dispatching workflow %s on branch %s with inputs %s",
            self.workflow_file,
            get_github_branch_name(),
            pprint.pformat(inputs),
        )
        success = workflow.create_dispatch(get_github_branch_name(), inputs=inputs)
        if success:
            await asyncio.sleep(2)
            runs = list(workflow.get_runs())

            for run in runs:
                if run.created_at.replace(tzinfo=timezone.utc) > trigger_time:
                    self.run = run
                    return True
        return False

    async def wait_for_completion(
        self, callback: Callable[["GitHubRun"], Awaitable[None]], timeout_minutes: int = 5
    ):
        if self.run is None:
            raise ValueError("Run needs to be triggered before a status check!")

        self.start_time = datetime.now(timezone.utc)
        timeout = timedelta(minutes=timeout_minutes)

        while True:
            try:
                # update run status
                self.run = run = self.repo.get_workflow_run(self.run_id)

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
                await asyncio.sleep(20)
            except TimeoutError:
                raise
            except Exception as e:
                logger.error(f"Error waiting for GitHub run {self.run_id}: {e}", exc_info=e)
                raise

    async def download_artifacts(self) -> dict:
        logger.info("Attempting to download artifacts for run %s", self.run_id)
        artifacts = self.run.get_artifacts()

        extracted = {}

        for artifact in artifacts:
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

                extracted[artifact.name] = artifact_dict
            else:
                raise RuntimeError(
                    f"Failed to download artifact {artifact.name}. "
                    f"Status code: {response.status_code}"
                )

        logger.info("Download artifacts for run %s: %s", self.run_id, list(extracted.keys()))
        return extracted
