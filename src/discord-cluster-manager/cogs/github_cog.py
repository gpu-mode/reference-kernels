import json

from cogs.submit_cog import ProgressReporter, SubmitCog
from consts import GitHubGPU, GPUType
from discord import app_commands
from github_runner import GitHubRun
from leaderboard_eval import amd_requirements, nvidia_requirements
from run_eval import CompileResult, FullResult, RunResult
from utils import setup_logging

logger = setup_logging()


class GitHubCog(SubmitCog):
    def __init__(self, bot):
        super().__init__(bot, name="GitHub", gpus=GitHubGPU)

    def _get_arch(self, gpu_type: app_commands.Choice[str]):
        return None

    async def _run_submission(
        self, config: dict, gpu_type: GPUType, status: ProgressReporter
    ) -> FullResult:
        selected_gpu = GPUType.AMD if gpu_type.value == "amd" else GPUType.NVIDIA

        lang = config["lang"]
        if lang == "cu" and selected_gpu == GPUType.AMD:
            # TODO implement HIP
            raise NotImplementedError("Cannot use CUDA runs with AMD GPUs")

        lang_name = {"py": "Python", "cu": "CUDA"}[lang]

        logger.info(f"Attempting to trigger GitHub action for {lang_name} on {selected_gpu.name}")

        workflow_file = selected_gpu.value
        run = GitHubRun(workflow_file)

        payload = json.dumps(config)

        inputs = {"payload": payload}
        if lang == "py":
            if selected_gpu == GPUType.NVIDIA:
                inputs["requirements"] = nvidia_requirements
            else:
                inputs["requirements"] = amd_requirements

        if not await run.trigger(inputs):
            raise RuntimeError("Failed to trigger GitHub Action. Please check the configuration.")

        await status.push("⏳ Waiting for workflow to start...")
        await run.wait_for_completion(lambda x: self.wait_callback(x, status))
        await status.update(f"Workflow [{run.run_id}]({run.html_url}) completed")
        await status.push("Downloading artifacts...")

        artifacts = await run.download_artifacts()
        logs = artifacts["run-result"]["result.json"].decode("utf-8")

        await status.update("Downloading artifacts... done")

        data = json.loads(logs)
        if "compile" in data and data["compile"] is not None:
            comp = CompileResult(**data["compile"])
        else:
            comp = None
        run = RunResult(**data["run"])
        return FullResult(success=True, error="", compile=comp, run=run)

    async def wait_callback(self, run: GitHubRun, status: ProgressReporter):
        await status.update(
            f"⏳ Workflow [{run.run_id}]({run.html_url}): {run.status} "
            f"({run.elapsed_time.total_seconds():.1f}s)"
        )
