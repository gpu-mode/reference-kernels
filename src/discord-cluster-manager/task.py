import copy
import dataclasses
import json
from pathlib import Path
from typing import Dict, Optional, Union

from consts import Language


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
        description: A description of the task.
            TODO use for a sticky message for the LBs channel
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
        templates: Template files for participants to download
        test_timeout, benchmark_timeout, ranked_timeout: Timeouts for running
            tests, benchmarks, and ranked submissions.

    """

    lang: Language
    files: dict[str, str]
    config: CudaTaskData | PythonTaskData
    description: str = ""
    libraries: list[str] = dataclasses.field(default_factory=list)
    tests: list[TestCaseType] = dataclasses.field(default_factory=list)
    test_timeout: int = 30
    benchmarks: list[TestCaseType] = dataclasses.field(default_factory=list)
    benchmark_timeout: int = 60
    ranked_timeout: int = 90
    templates: dict[str, str] = dataclasses.field(default_factory=dict)
    seed: Optional[int] = None

    @staticmethod
    def from_dict(data: dict):
        data_ = copy.copy(data)
        lang = Language(data["lang"])
        data_["lang"] = lang
        if lang == Language.Python:
            data_["config"] = PythonTaskData(**data["config"])
        else:
            data_["config"] = CudaTaskData(**data["config"])

        return LeaderboardTask(**data_)

    def to_dict(self) -> dict:
        raw = dataclasses.asdict(self)
        raw["lang"] = raw["lang"].value
        return raw

    def to_str(self):
        return json.dumps(self.to_dict(), sort_keys=True)

    @staticmethod
    def from_str(data: str):
        return LeaderboardTask.from_dict(json.loads(data))


def make_task(yaml_file: str | Path) -> LeaderboardTask:
    import yaml

    if Path(yaml_file).is_dir():
        yaml_file = Path(yaml_file) / "task.yml"

    with open(yaml_file) as f:
        raw = yaml.safe_load(f)

    root = Path(yaml_file).parent
    user_file_name = raw.get("user_file_name", None)

    # now, build file dict
    file_dict = {}
    for file_spec in raw["files"]:
        name = file_spec["name"]
        source = file_spec["source"]

        # handle special files
        if source == "@SUBMISSION@":
            assert user_file_name is None
            file_dict[name] = "@SUBMISSION@"
        else:
            file_dict[name] = (root / source).read_text()

    raw["files"] = file_dict

    # load template files
    templates = {}
    for lang, source in raw.get("templates", {}).items():
        assert lang in ["CUDA", "Python", "Triton"]
        templates[lang] = (root / source).read_text()
    raw["templates"] = templates

    return LeaderboardTask.from_dict(raw)


if __name__ == "__main__":
    print(json.dumps(make_task("task.yml").to_dict(), indent=4))
