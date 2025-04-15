import consts
import discord
from run_eval import CompileResult, EvalResult, RunResult
from utils import format_time


def _limit_length(text: str, maxlen: int):
    if len(text) > maxlen:
        return text[: maxlen - 6] + " [...]"
    else:
        return text


async def _send_split_log(thread: discord.Thread, partial_message: str, header: str, log: str):
    if len(partial_message) + len(log) + len(header) < 1900:
        partial_message += f"\n\n## {header}:\n"
        partial_message += f"```\n{log}```"
        return partial_message
    else:
        # send previous chunk
        if len(partial_message) > 0:
            await thread.send(partial_message)
        lines = log.splitlines()
        chunks = []
        partial_message = ""
        for line in lines:
            if len(partial_message) + len(line) < 1900:
                partial_message += line + "\n"
            else:
                if partial_message != "":
                    chunks.append(partial_message)
                partial_message = line

        if partial_message != "":
            chunks.append(partial_message)

        # now, format the chunks
        for i, chunk in enumerate(chunks):
            partial_message = f"\n\n## {header} ({i+1}/{len(chunks)}):\n"
            partial_message += f"```\n{_limit_length(chunk, 1900)}```"
            await thread.send(partial_message)

        return ""


async def _generate_compile_report(thread: discord.Thread, comp: CompileResult):
    message = ""
    if not comp.nvcc_found:
        message += "# Compilation failed\nNVCC could not be found.\n"
        message += "This indicates a bug in the runner configuration, _not in your code_.\n"
        message += "Please notify the server admins of this problem"
        await thread.send(message)
        return

    # ok, we found nvcc
    message += "# Compilation failed\n"
    message += "Command "
    message += f"```bash\n>{_limit_length(comp.command, 1000)}```\n"
    message += f"exited with code **{comp.exit_code}**."

    message = await _send_split_log(thread, message, "Compiler stderr", comp.stderr.strip())

    if len(comp.stdout.strip()) > 0:
        message = await _send_split_log(thread, message, "Compiler stdout", comp.stdout.strip())

    if len(message) != 0:
        await thread.send(message)


async def _generate_crash_report(thread: discord.Thread, run: RunResult):
    message = "# Running failed\n"
    message += "Command "
    message += f"```bash\n{_limit_length(run.command, 1000)}```\n"
    if run.exit_code == consts.ExitCode.TIMEOUT_EXPIRED:
        message += f"**timed out** after {float(run.duration):.2f} seconds."
    else:
        message += (
            f"exited with error code **{run.exit_code}** after {float(run.duration):.2f} seconds."
        )

    if len(run.stderr.strip()) > 0:
        message = await _send_split_log(thread, message, "Program stderr", run.stderr.strip())

    if len(run.stdout.strip()) > 0:
        message = await _send_split_log(thread, message, "Program stdout", run.stdout.strip())

    if len(message) != 0:
        await thread.send(message)


async def _generate_test_report(thread: discord.Thread, run: RunResult):
    message = "# Testing failed\n"
    message += "Command "
    message += f"```bash\n{_limit_length(run.command, 1000)}```\n"
    message += f"ran successfully in {run.duration:.2f} seconds, but did not pass all tests.\n"

    # Generate a test
    message = await _send_split_log(
        thread,
        message,
        "Test log",
        make_test_log(run),
    )

    if len(run.stderr.strip()) > 0:
        message = await _send_split_log(thread, message, "Program stderr", run.stderr.strip())

    if len(run.stdout.strip()) > 0:
        message = await _send_split_log(thread, message, "Program stdout", run.stdout.strip())

    if len(message) != 0:
        await thread.send(message)
    return


def make_short_report(runs: dict[str, EvalResult], full=True) -> list[str]:  # noqa: C901
    """
    Creates a minimalistic report for `runs`,
    returned as a list of status strings
    """
    any_compile = False
    result = []
    for r in runs.values():
        if r.compilation is not None:
            any_compile = True
            if not r.compilation.success:
                return ["❌ Compilation failed"]

    if any_compile:
        result.append("✅ Compilation successful")

    if "test" in runs:
        test_run = runs["test"].run
        if not test_run.success:
            result.append("❌ Running tests failed")
            return result
        elif not test_run.passed:
            result.append("❌ Testing failed")
            return result
        else:
            result.append("✅ Testing successful")
    elif full:
        result.append("❌ Tests missing")

    if "benchmark" in runs:
        bench_run = runs["benchmark"].run
        if not bench_run.success:
            result.append("❌ Running benchmarks failed")
            return result
        elif not bench_run.passed:
            result.append("❌ Benchmarking failed")
            return result
        else:
            result.append("✅ Benchmarking successful")
    elif full:
        result.append("❌ Benchmarks missing")

    if "leaderboard" in runs:
        lb_run = runs["leaderboard"].run
        if not lb_run.success:
            result.append("❌ Running leaderboard failed")
        elif not lb_run.passed:
            result.append("❌ Leaderboard run failed")
        else:
            result.append("✅ Leaderboard run successful")
    elif full:
        result.append("❌ Leaderboard missing")
    return result


def make_test_log(run: RunResult) -> str:
    test_log = []
    for i in range(len(run.result)):
        status = run.result.get(f"test.{i}.status", None)
        spec = run.result.get(f"test.{i}.spec", "<Error>")
        if status is None:
            break
        if status == "pass":
            test_log.append(f"✅ {spec}")
        elif status == "fail":
            test_log.append(f"❌ {spec}")
            error = run.result.get(f"test.{i}.error", "No error information available")
            if error:
                test_log.append(f"> {error}")
    if len(test_log) > 0:
        return str.join("\n", test_log)
    else:
        return "❗ Could not find any test cases"


def make_benchmark_log(run: RunResult) -> str:
    num_bench = int(run.result.get("benchmark-count", 0))

    def log_one(base_name):
        status = run.result.get(f"{base_name}.status")
        spec = run.result.get(f"{base_name}.spec")
        if status == "fail":
            bench_log.append(f"❌ {spec} failed testing:\n")
            bench_log.append(run.result.get(f"{base_name}.error"))
            return

        mean = run.result.get(f"{base_name}.mean")
        err = run.result.get(f"{base_name}.err")
        best = run.result.get(f"{base_name}.best")
        worst = run.result.get(f"{base_name}.worst")

        bench_log.append(f"{spec}")
        bench_log.append(f" ⏱ {format_time(mean, err)}")
        if best is not None and worst is not None:
            bench_log.append(f" ⚡ {format_time(best)} 🐌 {format_time(worst)}")

    bench_log = []
    for i in range(num_bench):
        log_one(f"benchmark.{i}")
        bench_log.append("")

    if len(bench_log) > 0:
        return "\n".join(bench_log)
    else:
        return "❗ Could not find any benchmarks"


async def generate_report(thread: discord.Thread, runs: dict[str, EvalResult]):  # noqa: C901
    message = ""

    if "test" in runs:
        test_run = runs["test"]

        if test_run.compilation is not None and not test_run.compilation.success:
            await _generate_compile_report(thread, test_run.compilation)
            return

        test_run = test_run.run

        if not test_run.success:
            await _generate_crash_report(thread, test_run)
            return

        if not test_run.passed:
            await _generate_test_report(thread, test_run)
            return
        else:
            num_tests = int(test_run.result.get("test-count", 0))
            for i in range(num_tests):
                status = test_run.result.get(f"test.{i}.status", None)
                if status is None:
                    break

            message = await _send_split_log(
                thread,
                message,
                "Tests",
                f"✅ Passed {num_tests}/{num_tests} tests",
            )

    if "benchmark" in runs:
        bench_run = runs["benchmark"]
        if bench_run.compilation is not None and not bench_run.compilation.success:
            await _generate_compile_report(thread, bench_run.compilation)
            return

        bench_run = bench_run.run
        if not bench_run.success:
            await _generate_crash_report(thread, bench_run)
            return

        message = await _send_split_log(
            thread,
            message,
            "Benchmarks",
            make_benchmark_log(bench_run),
        )

    if "leaderboard" in runs:
        bench_run = runs["leaderboard"]
        if bench_run.compilation is not None and not bench_run.compilation.success:
            await _generate_compile_report(thread, bench_run.compilation)
            return

        bench_run = bench_run.run
        if not bench_run.success:
            await _generate_crash_report(thread, bench_run)
            return

        message = await _send_split_log(
            thread,
            message,
            "Ranked Benchmark",
            make_benchmark_log(bench_run),
        )

    if "script" in runs:
        run = runs["script"]
        if run.compilation is not None and not run.compilation.success:
            await _generate_compile_report(thread, run.compilation)
            return

        run = run.run
        # OK, we were successful
        message += "# Success!\n"
        message += "Command "
        message += f"```bash\n{_limit_length(run.command, 1000)}```\n"
        message += f"ran successfully in {run.duration:.2} seconds.\n"

    if len(runs) == 1:
        run = next(iter(runs.values()))
        if len(run.run.stderr.strip()) > 0:
            message = await _send_split_log(
                thread, message, "Program stderr", run.run.stderr.strip()
            )

        if len(run.run.stdout.strip()) > 0:
            message = await _send_split_log(
                thread, message, "Program stdout", run.run.stdout.strip()
            )

    if len(message) != 0:
        await thread.send(message)


class MultiProgressReporter:
    def __init__(self, interaction: discord.Interaction, header: str):
        self.header = header
        self.runs = []
        self.interaction = interaction

    async def show(self):
        await self._update_message()

    def add_run(self, title: str) -> "RunProgressReporter":
        rpr = RunProgressReporterDiscord(self, self.interaction, title)
        self.runs.append(rpr)
        return rpr

    def make_message(self):
        formatted_runs = []
        for run in self.runs:
            formatted_runs.append(run.get_message())

        return str.join("\n\n", [f"# {self.header}"] + formatted_runs)

    async def _update_message(self):
        if self.interaction is None:
            return

        await self.interaction.edit_original_response(content=self.make_message(), view=None)


class RunProgressReporter:
    async def push(self, content: str | list[str]):
        raise NotImplementedError()

    async def update(self, new_content: str):
        raise NotImplementedError()

    async def update_title(self, new_title):
        raise NotImplementedError()

    async def generate_report(self, title: str, runs: dict[str, EvalResult]):
        raise NotImplementedError()


class RunProgressReporterDiscord(RunProgressReporter):
    def __init__(
        self,
        root: MultiProgressReporter,
        interaction: discord.Interaction,
        title: str,
    ):
        self.title = title
        self.lines = []
        self.root = root
        self.interaction = interaction

    async def push(self, content: str | list[str]):
        if isinstance(content, str):
            self.lines.append(f"> {content}")
        else:
            for line in content:
                self.lines.append(f"> {line}")
        await self._update_message()

    async def update(self, new_content: str):
        self.lines[-1] = f"> {new_content}"
        await self._update_message()

    async def update_title(self, new_title):
        self.title = new_title
        await self._update_message()

    async def _update_message(self):
        await self.root._update_message()

    def get_message(self):
        return str.join("\n", [f"**{self.title}**"] + self.lines)

    async def generate_report(self, title: str, runs: dict[str, EvalResult]):
        thread = await self.interaction.channel.create_thread(
            name=title,
            type=discord.ChannelType.private_thread,
            auto_archive_duration=1440,
        )
        await thread.add_user(self.interaction.user)
        await generate_report(thread, runs)
        await self.push(f"See results at {thread.jump_url}")


class RunProgressReporterAPI(RunProgressReporter):
    def __init__(self):
        self.title = ""
        self.lines = []

    async def push(self, content: str | list[str]):
        pass

    async def update(self, new_content: str):
        pass

    async def update_title(self, new_title):
        pass

    async def generate_report(self, title: str, runs: dict[str, EvalResult]):
        pass
