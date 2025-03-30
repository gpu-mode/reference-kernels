from enum import Enum
from typing import Type

from consts import GPUType
from report import RunProgressReporter


class Launcher:
    def __init__(self, name: str, gpus: Type[Enum]):
        self.name = name
        self.gpus = gpus

    async def run_submission(self, config: dict, gpu_type: GPUType, status: RunProgressReporter):
        raise NotImplementedError()
