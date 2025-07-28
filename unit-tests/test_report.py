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


# define helpers and  fixtures that create mock results
def sample_system_info() -> SystemInfo:
    return SystemInfo(
        gpu="NVIDIA RTX 4090", cpu="Intel i9-12900K", platform="Linux-5.15.0", torch="2.0.1+cu118"
    )


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


def create_eval_result() -> EvalResult:
    return EvalResult(
        start=datetime.datetime.now() - datetime.timedelta(minutes=5),
        end=datetime.datetime.now(),
        compilation=sample_compile_result(),
        run=sample_run_result(),
    )


@pytest.fixture
def sample_eval_result() -> EvalResult:
    return create_eval_result()


@pytest.fixture
def sample_full_result() -> FullResult:
    return FullResult(
        success=True, error="", system=sample_system_info(), runs={"test": create_eval_result()}
    )


################################################
#       Compilation report tests
################################################


def test_generate_compile_report_nvcc_not_found():
    compile_result = sample_compile_result()
    compile_result.success = False
    compile_result.nvcc_found = False
    compile_result.command = ""
    compile_result.exit_code = 127
    compile_result.stderr = "nvcc: command not found"
    compile_result.stdout = ""

    reporter = RunResultReport()
    _generate_compile_report(reporter, compile_result)

    assert len(reporter.data) == 1
    assert hasattr(reporter.data[0], "text")

    text = reporter.data[0].text
    assert "NVCC could not be found" in text
    assert "bug in the runner configuration" in text
    assert "notify the server admins" in text


def test_generate_compile_report_with_errors():
    compile_result = sample_compile_result()
    compile_result.success = False
    compile_result.nvcc_found = True
    compile_result.command = "nvcc -o test test.cu -arch=sm_75"
    compile_result.exit_code = 1
    compile_result.stderr = 'test.cu(15): error: identifier "invalid_function" is undefined'
    compile_result.stdout = "warning: deprecated feature used"

    reporter = RunResultReport()
    _generate_compile_report(reporter, compile_result)

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


def test_generate_compile_report_no_stdout():
    compile_result = sample_compile_result()
    compile_result.success = False
    compile_result.nvcc_found = True
    compile_result.command = "nvcc -o test test.cu"
    compile_result.exit_code = 1
    compile_result.stderr = "compilation error"
    compile_result.stdout = ""

    reporter = RunResultReport()
    _generate_compile_report(reporter, compile_result)

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


def test_short_fail_reason():
    run_result = sample_run_result()
    run_result.exit_code = consts.ExitCode.TIMEOUT_EXPIRED
    assert _short_fail_reason(run_result) == " (timeout)"

    run_result.exit_code = consts.ExitCode.CUDA_FAIL
    assert _short_fail_reason(run_result) == " (cuda api error)"

    # VALIDATE_FAIL means unit tests failed, which will be reported differently
    run_result.exit_code = consts.ExitCode.VALIDATE_FAIL
    assert _short_fail_reason(run_result) == ""

    run_result.exit_code = 42
    assert _short_fail_reason(run_result) == " (internal error 42)"


def test_make_short_report_compilation_failed(sample_eval_result: EvalResult):
    sample_eval_result.compilation.success = False
    runs = {"test": sample_eval_result}

    result = make_short_report(runs)
    assert result == ["❌ Compilation failed"]


def test_make_short_report_testing_failed(sample_eval_result: EvalResult):
    sample_eval_result.run.success = False
    sample_eval_result.run.exit_code = consts.ExitCode.TIMEOUT_EXPIRED
    runs = {"test": sample_eval_result}

    result = make_short_report(runs)
    assert result == ["✅ Compilation successful", "❌ Running tests failed (timeout)"]

    sample_eval_result.run.success = True
    sample_eval_result.run.passed = False
    sample_eval_result.run.exit_code = consts.ExitCode.VALIDATE_FAIL
    result = make_short_report(runs)
    assert result == ["✅ Compilation successful", "❌ Testing failed"]


def test_make_short_report_benchmarking_failed(sample_eval_result: EvalResult):
    sample_eval_result.run.success = False
    sample_eval_result.compilation = None
    sample_eval_result.run.exit_code = consts.ExitCode.CUDA_FAIL
    runs = {"benchmark": sample_eval_result}

    result = make_short_report(runs, full=False)
    assert result == ["❌ Running benchmarks failed (cuda api error)"]

    sample_eval_result.run.success = True
    sample_eval_result.run.passed = False
    sample_eval_result.run.exit_code = consts.ExitCode.VALIDATE_FAIL
    result = make_short_report(runs)
    assert result == ["❌ Tests missing", "❌ Benchmarking failed"]


def test_make_short_report_profiling_failed(sample_eval_result: EvalResult):
    sample_eval_result.run.success = False
    sample_eval_result.compilation = None
    sample_eval_result.run.exit_code = consts.ExitCode.PIPE_FAILED
    runs = {"profile": sample_eval_result}

    result = make_short_report(runs, full=False)
    assert result == ["❌ Running profile failed (internal error 111)"]

    sample_eval_result.run.success = True
    sample_eval_result.run.passed = False
    sample_eval_result.run.exit_code = consts.ExitCode.VALIDATE_FAIL
    result = make_short_report(runs)
    # TODO is this actually possible? Should profiling do **any** correctness testing?
    assert result == ["❌ Tests missing", "❌ Benchmarks missing", "❌ Profiling failed"]


def test_make_short_report_leaderboard_failed(sample_eval_result: EvalResult):
    sample_eval_result.run.success = False
    sample_eval_result.compilation = None
    sample_eval_result.run.exit_code = consts.ExitCode.TEST_SPEC
    runs = {"leaderboard": sample_eval_result}

    result = make_short_report(runs, full=False)
    assert result == ["❌ Running leaderboard failed (internal error 113)"]

    sample_eval_result.run.success = True
    sample_eval_result.run.passed = False
    sample_eval_result.run.exit_code = consts.ExitCode.VALIDATE_FAIL
    result = make_short_report(runs)
    # TODO is this actually possible? Should profiling do **any** correctness testing?
    assert result == ["❌ Tests missing", "❌ Benchmarks missing", "❌ Leaderboard run failed"]


def test_make_short_report_empty():
    result = make_short_report({})
    assert result == ["❌ Tests missing", "❌ Benchmarks missing", "❌ Leaderboard missing"]


def test_make_short_report_full_success():
    runs = {}
    for run_type in ["test", "benchmark", "profile", "leaderboard"]:
        runs[run_type] = EvalResult(
            start=datetime.datetime.now() - datetime.timedelta(minutes=5),
            end=datetime.datetime.now(),
            compilation=sample_compile_result(),
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


def test_make_short_report_missing_components():
    runs = {"test": create_eval_result()}

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


def test_make_test_log():
    log = make_test_log(sample_run_result())
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


def test_generate_system_info():
    info = generate_system_info(sample_system_info())

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
        stderr="Oh no a test failed!",
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
    from libkernelbot.report import Log, Text

    assert report.data == [
        Text(
            text="\n"
            "Running on:\n"
            "* GPU: `NVIDIA RTX 4090`\n"
            "* CPU: `Intel i9-12900K`\n"
            "* Platform: `Linux-5.15.0`\n"
            "* Torch: `2.0.1+cu118`\n"
        ),
        Text(
            text="# Testing failed\n"
            "Command ```bash\n"
            "python eval.py test```\n"
            "ran successfully in 2.10 seconds, but did not pass all tests.\n"
        ),
        Log(
            header="Test log",
            content="✅ Basic functionality\n"
            "❌ Edge case handling\n"
            "> Expected [1, 2, 3] but got [1, 2, 0]",
        ),
        Log(header="Program stderr", content="Oh no a test failed!"),
        Log(header="Program stdout", content="Running tests..."),
    ]


def test_generate_report_benchmark_failure(sample_full_result: FullResult):
    from libkernelbot.report import Log, Text

    sample_full_result.runs["benchmark"] = create_eval_result()
    report = generate_report(sample_full_result)
    assert report.data == [
        Text(
            text="\n"
            "Running on:\n"
            "* GPU: `NVIDIA RTX 4090`\n"
            "* CPU: `Intel i9-12900K`\n"
            "* Platform: `Linux-5.15.0`\n"
            "* Torch: `2.0.1+cu118`\n"
        ),
        Log(
            header="✅ Passed 3/3 tests",
            content="✅ Test addition\n"
            "> Addition works correctly\n"
            "✅ Test multiplication\n"
            "❌ Test division\n"
            "> Division by zero",
        ),
        Log(header="Benchmarks", content="❗ Could not find any benchmarks"),
    ]

    sample_full_result.runs["benchmark"].run.passed = False
    sample_full_result.runs["benchmark"].run.result = {
        "benchmark-count": "2",
        "benchmark.0.status": "pass",
        "benchmark.0.spec": "Basic functionality",
        "benchmark.0.mean": "10.5",
        "benchmark.0.err": "0.5",
        "benchmark.0.best": "9.8",
        "benchmark.0.worst": "15.2",
        "benchmark.1.status": "fail",
        "benchmark.1.spec": "Edge case handling",
        "benchmark.1.error": "Expected [1, 2, 3] but got [1, 2, 0]",
    }
    report = generate_report(sample_full_result)
    assert report.data == [
        Text(
            text="\n"
            "Running on:\n"
            "* GPU: `NVIDIA RTX 4090`\n"
            "* CPU: `Intel i9-12900K`\n"
            "* Platform: `Linux-5.15.0`\n"
            "* Torch: `2.0.1+cu118`\n"
        ),
        Log(
            header="✅ Passed 3/3 tests",
            content="✅ Test addition\n"
            "> Addition works correctly\n"
            "✅ Test multiplication\n"
            "❌ Test division\n"
            "> Division by zero",
        ),
        Log(
            header="Benchmarks",
            content="Basic functionality\n"
            " ⏱ 10.5 ± 0.50 ns\n"
            " ⚡ 9.80 ns 🐌 15.2 ns\n"
            "\n"
            "❌ Edge case handling failed testing:\n"
            "\n"
            "Expected [1, 2, 3] but got [1, 2, 0]\n",
        ),
    ]


def test_generate_report_leaderboard_failure(sample_full_result: FullResult):
    from libkernelbot.report import Log, Text

    sample_full_result.runs["leaderboard"] = create_eval_result()
    report = generate_report(sample_full_result)
    assert report.data == [
        Text(
            text="\n"
            "Running on:\n"
            "* GPU: `NVIDIA RTX 4090`\n"
            "* CPU: `Intel i9-12900K`\n"
            "* Platform: `Linux-5.15.0`\n"
            "* Torch: `2.0.1+cu118`\n"
        ),
        Log(
            header="✅ Passed 3/3 tests",
            content="✅ Test addition\n"
            "> Addition works correctly\n"
            "✅ Test multiplication\n"
            "❌ Test division\n"
            "> Division by zero",
        ),
        Log(header="Ranked Benchmark", content="❗ Could not find any benchmarks"),
    ]

    sample_full_result.runs["leaderboard"].run.success = False
    sample_full_result.runs["leaderboard"].run.exit_code = consts.ExitCode.TIMEOUT_EXPIRED
    sample_full_result.runs["leaderboard"].run.duration = 10.0

    report = generate_report(sample_full_result)
    assert report.data == [
        Text(
            text="\n"
            "Running on:\n"
            "* GPU: `NVIDIA RTX 4090`\n"
            "* CPU: `Intel i9-12900K`\n"
            "* Platform: `Linux-5.15.0`\n"
            "* Torch: `2.0.1+cu118`\n"
        ),
        Log(
            header="✅ Passed 3/3 tests",
            content="✅ Test addition\n"
            "> Addition works correctly\n"
            "✅ Test multiplication\n"
            "❌ Test division\n"
            "> Division by zero",
        ),
        Text(
            text="# Running failed\n"
            "Command ```bash\n"
            "./test```\n"
            "**timed out** after 10.00 seconds."
        ),
        Log(header="Program stdout", content="All tests passed"),
    ]


def test_generate_report_profile(sample_full_result: FullResult):
    sample_full_result.runs["profile"] = create_eval_result()
    sample_full_result.runs["profile"].run.result = {
        "benchmark-count": "1",
        "benchmark.0.spec": "Benchmark",
        "benchmark.0.report": base64.b64encode(b"Profile report", b"+*").decode("utf-8"),
    }
    report = generate_report(sample_full_result)
    from libkernelbot.report import Log, Text

    assert report.data == [
        Text(
            text="\n"
            "Running on:\n"
            "* GPU: `NVIDIA RTX 4090`\n"
            "* CPU: `Intel i9-12900K`\n"
            "* Platform: `Linux-5.15.0`\n"
            "* Torch: `2.0.1+cu118`\n"
        ),
        Log(
            header="✅ Passed 3/3 tests",
            content="✅ Test addition\n"
            "> Addition works correctly\n"
            "✅ Test multiplication\n"
            "❌ Test division\n"
            "> Division by zero",
        ),
        Log(header="Profiling", content="Benchmark\n\n  Profile report\n"),
    ]


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
