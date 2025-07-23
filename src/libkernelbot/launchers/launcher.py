from enum import Enum
from typing import Type

from libkernelbot.consts import GPU
from libkernelbot.report import RunProgressReporter


class Launcher:
    def __init__(self, name: str, gpus: Type[Enum]):
        self.name = name
        self.gpus = gpus

    async def run_submission(self, config: dict, gpu_type: GPU, status: RunProgressReporter):
        raise NotImplementedError()
