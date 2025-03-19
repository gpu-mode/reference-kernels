import datetime
import json

from cogs.submit_cog import SubmitCog
from consts import AMD_REQUIREMENTS, NVIDIA_REQUIREMENTS, GitHubGPU, GPUType
from discord import app_commands
from github_runner import GitHubRun
from report import RunProgressReporter
from run_eval import CompileResult, EvalResult, FullResult, RunResult, SystemInfo
from utils import setup_logging

logger = setup_logging()


class GitHubCog(SubmitCog):
    def __init__(self, bot):
        super().__init__(bot, name="GitHub", gpus=GitHubGPU)

    def _get_arch(self, gpu_type: app_commands.Choice[str]):
        return None

    async def _run_submission(
        self, config: dict, gpu_type: GPUType, status: RunProgressReporter
    ) -> FullResult:
        selected_gpu = GPUType.AMD if gpu_type.value == "amd" else GPUType.NVIDIA

        lang = config["lang"]
        if lang == "cu" and selected_gpu == GPUType.AMD:
            # TODO implement HIP
            raise NotImplementedError("Cannot use CUDA runs with AMD GPUs")

        lang_name = {"py": "Python", "cu": "CUDA"}[lang]

        if selected_gpu == GPUType.AMD:
            gpu_name = config.get("gpu", "mi300")
            runner_name = {"mi250": "amdgpu-mi250-x86-64", "mi300": "amdgpu-mi300-x86-64"}[gpu_name]

        logger.info(f"Attempting to trigger GitHub action for {lang_name} on {selected_gpu.name}")
        if selected_gpu == GPUType.AMD:
            logger.info(f"Running on {gpu_name} amd gpu")

        workflow_file = selected_gpu.value
        run = GitHubRun(workflow_file)

        payload = json.dumps(config)

        inputs = {"payload": payload}
        if lang == "py":
            if selected_gpu == GPUType.NVIDIA:
                inputs["requirements"] = NVIDIA_REQUIREMENTS
            else:
                inputs["requirements"] = AMD_REQUIREMENTS
                inputs["runner"] = runner_name
        if not await run.trigger(inputs):
            raise RuntimeError("Failed to trigger GitHub Action. Please check the configuration.")

        await status.push("⏳ Waiting for workflow to start...")
        await run.wait_for_completion(lambda x: self.wait_callback(x, status))
        await status.update(f"Workflow [{run.run_id}]({run.html_url}) completed")
        await status.push("Downloading artifacts...")

        artifacts = await run.download_artifacts()
        if "run-result" not in artifacts:
            logger.error("Could not find `run-result` among artifacts: %s", artifacts.keys())
            await status.push("Downloading artifacts...  failed")
            return FullResult(
                success=False, error="Could not download artifacts", runs={}, system=SystemInfo()
            )

        logs = artifacts["run-result"]["result.json"].decode("utf-8")

        await status.update("Downloading artifacts... done")

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

    async def wait_callback(self, run: GitHubRun, status: RunProgressReporter):
        await status.update(
            f"⏳ Workflow [{run.run_id}]({run.html_url}): {run.status} "
            f"({run.elapsed_time.total_seconds():.1f}s)"
        )
