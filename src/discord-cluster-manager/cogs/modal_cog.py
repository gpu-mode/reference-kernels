import asyncio

import modal
from cogs.submit_cog import ProgressReporter, SubmitCog
from consts import GPU_TO_SM, MODAL_CUDA_INCLUDE_DIRS, GPUType, ModalGPU
from discord import app_commands
from run_eval import FullResult
from utils import setup_logging

logger = setup_logging()


class ModalCog(SubmitCog):
    def __init__(self, bot):
        super().__init__(bot, "Modal", gpus=ModalGPU)

    def _get_arch(self, gpu_type: app_commands.Choice[str]):
        return GPU_TO_SM[gpu_type.value.upper()]

    async def _run_submission(
        self, config: dict, gpu_type: GPUType, status: ProgressReporter
    ) -> FullResult:
        loop = asyncio.get_event_loop()
        if config["lang"] == "cu":
            config["include_dirs"] = config.get("include_dirs", []) + MODAL_CUDA_INCLUDE_DIRS
        func_type = "pytorch" if config["lang"] == "py" else "cuda"
        func_name = f"run_{func_type}_script_{gpu_type.value.lower()}"

        logger.info(f"Starting modal run using {func_name}")

        await status.push("⏳ Waiting for available GPU...")

        result = await loop.run_in_executor(
            None,
            lambda: modal.Function.lookup("discord-bot-runner", func_name).remote(config=config),
        )

        await status.update("  Waiting for available GPU... Done")

        return result
