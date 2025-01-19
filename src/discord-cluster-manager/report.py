import pprint

import discord
from run_eval import FullResult


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
        await thread.send(partial_message)
        lines = log.splitlines()
        chunks = []
        partial_message = ""
        for line in lines:
            if len(partial_message) + len(line) < 1900:
                partial_message += line + "\n"
            else:
                chunks.append(partial_message)
                partial_message = line

        if partial_message != "":
            chunks.append(partial_message)

        # now, format the chunks
        for i, chunk in enumerate(chunks):
            partial_message += f"\n\n## {header} ({i+1}/{len(chunks)}):\n"
            partial_message += f"```\n{_limit_length(chunk, 1900)}```"
            await thread.send(partial_message)

        return ""


async def generate_report(thread: discord.Thread, result: FullResult, has_tests: bool):  # noqa: C901
    message = ""
    if not result.success:
        message += "# Failure\n"
        message += result.error
        await thread.send(message)
        return

    comp = result.compile
    run = result.run

    message = ""
    if comp is not None and not comp.success:
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

        return

    if not run.success:
        message += "# Running failed\n"
        message += "Command "
        message += f"```bash\n{_limit_length(run.command, 1000)}```\n"
        message += f"exited with error code **{run.exit_code}** after {run.duration:.2} seconds."

        if len(run.stderr.strip()) > 0:
            message = await _send_split_log(thread, message, "Program stderr", run.stderr.strip())

        if len(run.stdout.strip()) > 0:
            message = await _send_split_log(thread, message, "Program stdout", run.stdout.strip())

        if len(message) != 0:
            await thread.send(message)

        return

    if has_tests:
        if not run.passed:
            message += "# Testing failed\n"
            message += "Command "
            message += f"```bash\n{_limit_length(run.command, 1000)}```\n"
            message += (
                f"ran successfully in {run.duration:.2} seconds, but did not pass all tests.\n"
            )

            if len(run.stderr.strip()) > 0:
                message = await _send_split_log(
                    thread, message, "Program stderr", run.stderr.strip()
                )

            if len(run.stdout.strip()) > 0:
                message = await _send_split_log(
                    thread, message, "Program stdout", run.stdout.strip()
                )

            if len(message) != 0:
                await thread.send(message)

            # TODO dedicated "error" entry in our results that gets
            # populated by check_implementation
            return

    # OK, we were successful
    message += "# Success!\n"
    message += "Command "
    message += f"```bash\n{_limit_length(run.command, 1000)}```\n"
    message += f"ran successfully in {run.duration:.2} seconds.\n"

    if has_tests:
        message = await _send_split_log(thread, message, "Result", pprint.pformat(run.result))

    if len(run.stderr.strip()) > 0:
        message = await _send_split_log(thread, message, "Program stderr", run.stderr.strip())

    if len(run.stdout.strip()) > 0:
        message = await _send_split_log(thread, message, "Program stdout", run.stdout.strip())

    if len(message) != 0:
        await thread.send(message)
