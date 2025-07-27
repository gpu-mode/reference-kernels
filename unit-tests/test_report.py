import base64
import datetime
from unittest.mock import Mock

import pytest

from libkernelbot import consts
from libkernelbot.report import (
    RunResultReport,
    _generate_compile_report,
    _short_fail_reason,
    generate_report,
    generate_system_info,
    make_benchmark_log,
    make_profile_log,
    make_short_report,
    make_test_log,
)
from libkernelbot.run_eval import CompileResult, EvalResult, FullResult, RunResult, SystemInfo


# define fixtures that create mock results
@pytest.fixture
def sample_system_info() -> SystemInfo:
    return SystemInfo(
        gpu="NVIDIA RTX 4090", cpu="Intel i9-12900K", platform="Linux-5.15.0", torch="2.0.1+cu118"
    )


@pytest.fixture
def sample_compile_result() -> CompileResult:
    return CompileResult(
        success=True,
        nvcc_found=True,
        nvcc_version="11.8",
        command="nvcc -o test test.cu",
        exit_code=0,
        stdout="",
        stderr="",
    )


@pytest.fixture
def sample_run_result() -> RunResult:
    return RunResult(
        success=True,
        passed=True,
        command="./test",
        exit_code=0,
        duration=1.5,
        stdout="All tests passed",
        stderr="",
        result={
            "test-count": "3",
            "test.0.status": "pass",
            "test.0.spec": "Test addition",
            "test.0.message": "Addition works correctly",
            "test.1.status": "pass",
            "test.1.spec": "Test multiplication",
            "test.2.status": "fail",
            "test.2.spec": "Test division",
            "test.2.error": "Division by zero",
        },
    )


@pytest.fixture
def sample_eval_result(
    sample_compile_result: CompileResult, sample_run_result: RunResult
) -> EvalResult:
    return EvalResult(
        start=datetime.datetime.now() - datetime.timedelta(minutes=5),
        end=datetime.datetime.now(),
        compilation=sample_compile_result,
        run=sample_run_result,
    )


@pytest.fixture
def sample_full_result(
    sample_system_info: SystemInfo, sample_eval_result: EvalResult
) -> FullResult:
    return FullResult(
        success=True, error="", system=sample_system_info, runs={"test": sample_eval_result}
    )


################################################
#       Compilation report tests
################################################


def test_generate_compile_report_nvcc_not_found(sample_compile_result: CompileResult):
    sample_compile_result.success = False
    sample_compile_result.nvcc_found = False
    sample_compile_result.command = ""
    sample_compile_result.exit_code = 127
    sample_compile_result.stderr = "nvcc: command not found"
    sample_compile_result.stdout = ""

    reporter = RunResultReport()
    _generate_compile_report(reporter, sample_compile_result)

    assert len(reporter.data) == 1
    assert hasattr(reporter.data[0], "text")

    text = reporter.data[0].text
    assert "NVCC could not be found" in text
    assert "bug in the runner configuration" in text
    assert "notify the server admins" in text


def test_generate_compile_report_with_errors(sample_compile_result: CompileResult):
    sample_compile_result.success = False
    sample_compile_result.nvcc_found = True
    sample_compile_result.command = "nvcc -o test test.cu -arch=sm_75"
    sample_compile_result.exit_code = 1
    sample_compile_result.stderr = 'test.cu(15): error: identifier "invalid_function" is undefined'
    sample_compile_result.stdout = "warning: deprecated feature used"

    reporter = RunResultReport()
    _generate_compile_report(reporter, sample_compile_result)

    # Should have compilation text + stderr log + stdout log
    assert len(reporter.data) == 3

    # Check compilation failure message
    assert hasattr(reporter.data[0], "text")
    text = reporter.data[0].text
    assert "Compilation failed" in text
    assert "nvcc -o test test.cu -arch=sm_75" in text
    assert "exited with code **1**" in text

    # Check stderr and stdout logs
    assert hasattr(reporter.data[1], "header")
    assert reporter.data[1].header == "Compiler stderr"
    assert 'identifier "invalid_function" is undefined' in reporter.data[1].content

    assert hasattr(reporter.data[2], "header")
    assert reporter.data[2].header == "Compiler stdout"
    assert "warning: deprecated feature used" in reporter.data[2].content


def test_generate_compile_report_no_stdout(sample_compile_result: CompileResult):
    sample_compile_result.success = False
    sample_compile_result.nvcc_found = True
    sample_compile_result.command = "nvcc -o test test.cu"
    sample_compile_result.exit_code = 1
    sample_compile_result.stderr = "compilation error"
    sample_compile_result.stdout = ""

    reporter = RunResultReport()
    _generate_compile_report(reporter, sample_compile_result)

    # Should have compilation text + stderr log (no stdout log)
    assert len(reporter.data) == 2

    assert hasattr(reporter.data[0], "text")
    assert "Compilation failed" in reporter.data[0].text

    assert hasattr(reporter.data[1], "header")
    assert reporter.data[1].header == "Compiler stderr"
    assert "compilation error" in reporter.data[1].content


################################################
#       Short report tests
################################################


def test_short_fail_reason(sample_run_result: RunResult):
    sample_run_result.exit_code = consts.ExitCode.TIMEOUT_EXPIRED
    assert _short_fail_reason(sample_run_result) == " (timeout)"

    sample_run_result.exit_code = consts.ExitCode.CUDA_FAIL
    assert _short_fail_reason(sample_run_result) == " (cuda api error)"

    # VALIDATE_FAIL means unit tests failed, which will be reported differently
    sample_run_result.exit_code = consts.ExitCode.VALIDATE_FAIL
    assert _short_fail_reason(sample_run_result) == ""

    sample_run_result.exit_code = 42
    assert _short_fail_reason(sample_run_result) == " (internal error 42)"


def test_make_short_report_compilation_failed(sample_eval_result: EvalResult):
    sample_eval_result.compilation.success = False
    runs = {"test": sample_eval_result}

    result = make_short_report(runs)
    assert result == ["❌ Compilation failed"]


def test_make_short_report_full_success(sample_compile_result: CompileResult):
    runs = {}
    for run_type in ["test", "benchmark", "profile", "leaderboard"]:
        runs[run_type] = EvalResult(
            start=datetime.datetime.now() - datetime.timedelta(minutes=5),
            end=datetime.datetime.now(),
            compilation=sample_compile_result,
            run=RunResult(
                success=True,
                passed=True,
                command="./test",
                exit_code=0,
                duration=1.5,
                stdout="",
                stderr="",
                result={},
            ),
        )

    result = make_short_report(runs, full=True)
    expected = [
        "✅ Compilation successful",
        "✅ Testing successful",
        "✅ Benchmarking successful",
        "✅ Profiling successful",
        "✅ Leaderboard run successful",
    ]
    assert result == expected


def test_make_short_report_missing_components(sample_eval_result: EvalResult):
    runs = {"test": sample_eval_result}

    result = make_short_report(runs, full=True)
    expected = [
        "✅ Compilation successful",
        "✅ Testing successful",
        "❌ Benchmarks missing",
        "❌ Leaderboard missing",
    ]
    assert result == expected


################################################
#    Test, Benchmark, Profile reporting
################################################


def test_make_test_log(sample_run_result: RunResult):
    log = make_test_log(sample_run_result)
    expected_lines = [
        "✅ Test addition",
        "> Addition works correctly",
        "✅ Test multiplication",
        "❌ Test division",
        "> Division by zero",
    ]
    assert log == "\n".join(expected_lines)


def test_make_test_log_no_tests():
    run = Mock()
    run.result = {}

    log = make_test_log(run)
    assert log == "❗ Could not find any test cases"


def test_make_benchmark_log():
    run = Mock()
    run.result = {
        "benchmark-count": "2",
        "benchmark.0.status": "pass",
        "benchmark.0.spec": "Matrix multiplication",
        "benchmark.0.mean": "1.5",
        "benchmark.0.err": "0.1",
        "benchmark.0.best": "1.3",
        "benchmark.0.worst": "1.8",
        "benchmark.1.status": "fail",
        "benchmark.1.spec": "Vector addition",
        "benchmark.1.error": "Timeout occurred",
    }

    log = make_benchmark_log(run)

    assert "Matrix multiplication" in log
    assert "Vector addition" in log
    assert "❌ Vector addition failed testing:" in log
    assert "Timeout occurred" in log


def test_make_benchmark_log_no_benchmarks():
    run = Mock()
    run.result = {"benchmark-count": "0"}

    log = make_benchmark_log(run)
    assert log == "❗ Could not find any benchmarks"


def test_make_profile_log():
    profile_data = "Profile line 1\nProfile line 2"
    encoded_data = base64.b64encode(profile_data.encode("utf-8"), b"+*").decode("utf-8")

    run = Mock()
    run.result = {
        "benchmark-count": "1",
        "benchmark.0.spec": "Matrix multiplication profile",
        "benchmark.0.report": encoded_data,
    }

    log = make_profile_log(run)

    assert "Matrix multiplication profile" in log
    assert "  Profile line 1" in log
    assert "  Profile line 2" in log


def test_make_profile_log_no_data():
    run = Mock()
    run.result = {"benchmark-count": "0"}

    log = make_profile_log(run)
    assert log == "❗ Could not find any profiling data"


def test_generate_system_info(sample_system_info: SystemInfo):
    info = generate_system_info(sample_system_info)

    expected_parts = ["NVIDIA RTX 4090", "Intel i9-12900K", "Linux-5.15.0", "2.0.1+cu118"]

    for part in expected_parts:
        assert part in info


################################################
#    Full report
################################################


def test_generate_report_successful_test(sample_full_result: FullResult):
    report = generate_report(sample_full_result)

    assert len(report.data) >= 2
    assert any("NVIDIA RTX 4090" in item.text for item in report.data if hasattr(item, "text"))


def test_generate_report_compilation_failure(sample_full_result: FullResult):
    sample_full_result.runs["test"].compilation.success = False

    report = generate_report(sample_full_result)

    # Should return early with compilation failure using _generate_compile_report
    text_items = [item.text for item in report.data if hasattr(item, "text")]
    assert any("Compilation failed" in text for text in text_items)


def test_generate_report_runtime_crash(sample_full_result: FullResult):
    sample_full_result.runs["test"].run = RunResult(
        success=False,
        passed=False,
        command="./test_kernel",
        exit_code=consts.ExitCode.CUDA_FAIL,
        duration=0.5,
        stdout="",
        stderr="CUDA error: invalid device function",
        result={},
    )

    report = generate_report(sample_full_result)

    # Should have system info + crash message + stderr log
    assert len(report.data) == 3

    text_items = [item.text for item in report.data if hasattr(item, "text")]
    crash_text = next(text for text in text_items if "Running failed" in text)

    assert "./test_kernel" in crash_text
    assert f"exited with error code **{consts.ExitCode.CUDA_FAIL}**" in crash_text
    assert "after 0.50 seconds" in crash_text

    # Check stderr log
    log_items = [item for item in report.data if hasattr(item, "header")]
    stderr_log = next(item for item in log_items if item.header == "Program stderr")
    assert "CUDA error: invalid device function" in stderr_log.content


def test_generate_report_test_failure(sample_full_result: FullResult):
    sample_full_result.runs["test"].run = RunResult(
        success=True,
        passed=False,
        command="python eval.py test",
        exit_code=consts.ExitCode.VALIDATE_FAIL,
        duration=2.1,
        stdout="Running tests...",
        stderr="",
        result={
            "test-count": "2",
            "test.0.status": "pass",
            "test.0.spec": "Basic functionality",
            "test.1.status": "fail",
            "test.1.spec": "Edge case handling",
            "test.1.error": "Expected [1, 2, 3] but got [1, 2, 0]",
        },
    )

    report = generate_report(sample_full_result)

    # Should have system info + test failure text + test log + stdout log
    assert len(report.data) == 4

    text_items = [item.text for item in report.data if hasattr(item, "text")]
    test_text = next(text for text in text_items if "Testing failed" in text)

    assert "python eval.py test" in test_text
    assert "ran successfully in 2.10 seconds" in test_text
    assert "did not pass all tests" in test_text

    # Check test log contains failure details
    log_items = [item for item in report.data if hasattr(item, "header")]
    test_log = next(item for item in log_items if item.header == "Test log")
    assert "✅ Basic functionality" in test_log.content
    assert "❌ Edge case handling" in test_log.content
    assert "Expected [1, 2, 3] but got [1, 2, 0]" in test_log.content


def test_run_result_report():
    reporter = RunResultReport()

    reporter.add_text("Test message")
    reporter.add_log("Test Header", "Test log content")

    assert len(reporter.data) == 2
    assert hasattr(reporter.data[0], "text")
    assert reporter.data[0].text == "Test message"
    assert hasattr(reporter.data[1], "header")
    assert reporter.data[1].header == "Test Header"
    assert reporter.data[1].content == "Test log content"
