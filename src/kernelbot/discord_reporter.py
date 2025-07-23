import discord
from discord_utils import _send_split_log

from libkernelbot.report import (
    Log,
    MultiProgressReporter,
    RunProgressReporter,
    RunResultReport,
    Text,
)


class MultiProgressReporterDiscord(MultiProgressReporter):
    def __init__(self, interaction: discord.Interaction):
        self.header = ""
        self.runs = []
        self.interaction = interaction

    async def show(self, title: str):
        self.header = title
        await self._update_message()

    def add_run(self, title: str) -> "RunProgressReporterDiscord":
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


class RunProgressReporterDiscord(RunProgressReporter):
    def __init__(
        self,
        root: MultiProgressReporterDiscord,
        interaction: discord.Interaction,
        title: str,
    ):
        super().__init__(title=title)
        self.root = root
        self.interaction = interaction

    async def _update_message(self):
        await self.root._update_message()

    async def display_report(self, title: str, report: RunResultReport):
        thread = await self.interaction.channel.create_thread(
            name=title,
            type=discord.ChannelType.private_thread,
            auto_archive_duration=1440,
        )
        await thread.add_user(self.interaction.user)
        message = ""
        for part in report.data:
            if isinstance(part, Text):
                if len(message) + len(part.text) > 1900:
                    await thread.send(message)
                    message = ""
                message += part.text
            elif isinstance(part, Log):
                message = await _send_split_log(thread, message, part.header, part.content)

        if len(message) > 0:
            await thread.send(message)

        await self.push(f"See results at {thread.jump_url}")
