import datetime
from typing import Any, Dict

from run_eval import CompileResult, EvalResult, FullResult, RunResult, SystemInfo


def deserialize_full_result(data: Dict[str, Any]) -> FullResult:
    """
    Deserialize a JSON dictionary into a FullResult dataclass object.

    Args:
        data: Dictionary from response.json()

    Returns:
        FullResult object with all nested components properly deserialized
    """
    # Deserialize SystemInfo
    system_data = data.get("system", {})
    system_info = SystemInfo(
        gpu=system_data.get("gpu", ""),
        cpu=system_data.get("cpu", ""),
        platform=system_data.get("platform", ""),
        torch=system_data.get("torch", ""),
    )

    # Deserialize runs (dict of EvalResults)
    runs_data = data.get("runs", {})
    runs = {}
    for run_key, run_value in runs_data.items():
        runs[run_key] = deserialize_eval_result(run_value)

    # Create the FullResult dataclass
    return FullResult(
        success=data.get("success", False),
        error=data.get("error", ""),
        system=system_info,
        runs=runs,
    )


def deserialize_eval_result(data: Dict[str, Any]) -> EvalResult:
    """
    Deserialize a dictionary into an EvalResult object.

    Args:
        data: Dictionary representing an EvalResult

    Returns:
        EvalResult object with all nested components properly deserialized
    """
    # Parse datetime strings
    start = (
        datetime.datetime.fromisoformat(data.get("start"))
        if data.get("start")
        else datetime.datetime.now()
    )
    end = (
        datetime.datetime.fromisoformat(data.get("end"))
        if data.get("end")
        else datetime.datetime.now()
    )

    # Deserialize CompileResult
    compilation_data = data.get("compilation")
    compilation = None
    if compilation_data:
        compilation = CompileResult(
            nvcc_found=compilation_data.get("nvcc_found", False),
            nvcc_version=compilation_data.get("nvcc_version", ""),
            success=compilation_data.get("success", False),
            command=compilation_data.get("command", ""),
            stdout=compilation_data.get("stdout", ""),
            stderr=compilation_data.get("stderr", ""),
            exit_code=compilation_data.get("exit_code", 1),
        )

    # Deserialize RunResult
    run_data = data.get("run")
    run = None
    if run_data:
        run = RunResult(
            success=run_data.get("success", False),
            passed=run_data.get("passed", False),
            command=run_data.get("command", ""),
            stdout=run_data.get("stdout", ""),
            stderr=run_data.get("stderr", ""),
            exit_code=run_data.get("exit_code", 1),
            duration=run_data.get("duration", 0.0),
            result=run_data.get("result", {}),
        )

    return EvalResult(start=start, end=end, compilation=compilation, run=run)
