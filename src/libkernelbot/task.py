import copy
import dataclasses
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import yaml

from libkernelbot.consts import Language, RankCriterion, SubmissionMode
from libkernelbot.utils import KernelBotError


@dataclasses.dataclass
class CudaTaskData:
    sources: list[str]
    include_dirs: list[str] = dataclasses.field(default_factory=list)
    defines: dict[str, str] = dataclasses.field(default_factory=dict)
    compile_flags: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class PythonTaskData:
    main: str


TestCaseType = Dict[str, Union[int, str]]


@dataclasses.dataclass
class LeaderboardTask:
    """
    Dataclass containing the definition of a task for the leaderboard

    Attributes:
        lang: Programming language of this task. Specifies the type of
            the `data` attribute.
        files: Dictionary containing a mapping of file names to file
            contents. Contents '@SUBMISSION@' get replaced with the
            submitted file before sending to the runner.
        libraries: List of string identifiers for libraries available
            (and potentially required) for this task. How these strings
            are interpreted is up to the individual runner.
        config: Language-specific task definition.
        tests: List of test case specifications. Each test case is specified
            as a dict mapping function argument names to their values.
        benchmarks: List of benchmark specifications (same format as tests)
        test_timeout, benchmark_timeout, ranked_timeout: Timeouts for running
            tests, benchmarks, and ranked submissions.

    """

    lang: Language
    files: dict[str, str]
    config: CudaTaskData | PythonTaskData
    libraries: list[str] = dataclasses.field(default_factory=list)
    tests: list[TestCaseType] = dataclasses.field(default_factory=list)
    test_timeout: int = 180
    benchmarks: list[TestCaseType] = dataclasses.field(default_factory=list)
    benchmark_timeout: int = 180
    ranked_timeout: int = 180
    ranking_by: RankCriterion = RankCriterion.LAST
    seed: Optional[int] = None

    def __post_init__(self):
        if self.lang == Language.Python and not isinstance(self.config, PythonTaskData):
            raise TypeError("Python language requires PythonTaskData config")
        if self.lang == Language.CUDA and not isinstance(self.config, CudaTaskData):
            raise TypeError("CUDA language requires CudaTaskData config")

    @classmethod
    def from_dict(cls, data: dict):
        data_ = copy.copy(data)
        lang = Language(data["lang"])
        criterion = RankCriterion(data.get("ranking_by", RankCriterion.LAST))
        data_["lang"] = lang
        data_["ranking_by"] = criterion
        if lang == Language.Python:
            data_["config"] = PythonTaskData(**data["config"])
        else:
            data_["config"] = CudaTaskData(**data["config"])

        return cls(**data_)

    def to_dict(self) -> dict:
        raw = dataclasses.asdict(self)
        raw["lang"] = raw["lang"].value
        raw["ranking_by"] = raw["ranking_by"].value
        return raw

    def to_str(self):
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_str(cls, data: str):
        return cls.from_dict(json.loads(data))


@dataclasses.dataclass
class LeaderboardDefinition:
    """
    LeaderboardDefinition extends LeaderboardTask with additional (meta)data
    that is not directly required for running a submission.

    description: A description of the task.
    TODO use for a sticky message for the LBs channel
    templates: Template files for participants to download
    """

    task: LeaderboardTask
    description: str = ""
    templates: dict[str, str] = dataclasses.field(default_factory=dict)


def make_task_definition(yaml_file: str | Path) -> LeaderboardDefinition:
    if Path(yaml_file).is_dir():
        yaml_file = Path(yaml_file) / "task.yml"

    try:
        with open(yaml_file) as f:
            raw = yaml.safe_load(f)
    except yaml.parser.ParserError as E:
        logging.exception("Error loading task.yml", exc_info=E)
        raise KernelBotError(f"Error loading task.yml: {E}") from E

    root = Path(yaml_file).parent

    # now, build file dict
    file_dict = {}
    for file_spec in raw["files"]:
        name = file_spec["name"]
        source = file_spec["source"]

        # handle special files
        if source == "@SUBMISSION@":
            file_dict[name] = "@SUBMISSION@"
        else:
            file_dict[name] = (root / source).read_text()

    raw["files"] = file_dict

    # load template files
    templates = {}
    for lang, source in raw.get("templates", {}).items():
        assert lang in ["CUDA", "Python", "Triton", "HIP"]
        templates[lang] = (root / source).read_text()

    del raw["templates"]
    description = raw["description"]
    del raw["description"]
    task = LeaderboardTask.from_dict(raw)
    return LeaderboardDefinition(task=task, templates=templates, description=description)


def build_task_config(
    task: LeaderboardTask = None,
    submission_content: str = None,
    arch: str = None,
    mode: SubmissionMode = None,
) -> dict:
    all_files = {}
    for n, c in task.files.items():
        if c == "@SUBMISSION@":
            all_files[n] = submission_content
        else:
            all_files[n] = c

    common = {
        "lang": task.lang.value,
        "arch": arch,
        "benchmarks": task.benchmarks,
        "tests": task.tests,
        "mode": mode.value,
        "test_timeout": task.test_timeout,
        "benchmark_timeout": task.benchmark_timeout,
        "ranked_timeout": task.ranked_timeout,
        "ranking_by": task.ranking_by.value,
        "seed": task.seed,
    }

    if task.lang == Language.Python:
        return {
            "main": task.config.main,
            "sources": all_files,
            **common,
        }
    else:
        sources = {}
        headers = {}
        for f in all_files:
            if f in task.config.sources:
                sources[f] = all_files[f]
            else:
                headers[f] = all_files[f]

        return {
            "sources": sources,
            "headers": headers,
            "defines": task.config.defines,
            "compile_flags": task.config.compile_flags,
            "include_dirs": task.config.include_dirs,
            **common,
        }
