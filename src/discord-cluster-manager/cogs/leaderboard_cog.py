import asyncio
from datetime import datetime
from io import StringIO
from typing import TYPE_CHECKING, Callable, List, Optional

import discord
from consts import (
    SubmissionMode,
    get_gpu_by_name,
)
from discord import app_commands
from discord.ext import commands, tasks
from leaderboard_db import leaderboard_name_autocomplete
from task import LeaderboardTask
from ui.misc import GPUSelectionView
from ui.table import create_table
from utils import (
    LeaderboardItem,
    get_user_from_id,
    send_discord_message,
    setup_logging,
)

if TYPE_CHECKING:
    from ..bot import ClusterBot

logger = setup_logging()


class LeaderboardSubmitCog(app_commands.Group):
    def __init__(self, bot: "ClusterBot"):
        super().__init__(name="submit", description="Submit to leaderboard")
        self.bot = bot

    async def async_submit_cog_job(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        script: discord.Attachment,
        command: Callable,
        task: LeaderboardTask,
        submission_content,
        gpu: app_commands.Choice[str],
        runner_name: str,
        mode: SubmissionMode,
    ):
        discord_thread, result = await command(
            interaction,
            script,
            gpu,
            task=task,
            mode=mode,
        )

        # no point going further if this already failed
        if discord_thread is None:
            return -1

        if mode == SubmissionMode.LEADERBOARD:
            pass
            # public leaderboard run
        elif mode == SubmissionMode.PRIVATE:
            pass
            # private leaderboard run
        else:
            return 0

        try:
            if result.success:
                user_id = (
                    interaction.user.global_name
                    if interaction.user.nick is None
                    else interaction.user.nick
                )

                if result.runs["test"].run.result["check"] != "pass":
                    await discord_thread.send(
                        f"Ran on {gpu.name} using {runner_name} runners!\n"
                        + f"Leaderboard '{leaderboard_name}'.\n"
                        + f"Submission title: {script.filename}.\n"
                        + f"Submission user: {user_id}.\n"
                    )
                    return

                # TODO: Make this more flexible, not just functional
                score = 0.0
                num_benchmarks = int(result.runs["benchmark"].run.result["benchmark-count"])
                for i in range(num_benchmarks):
                    score += float(result.runs["benchmark"].run.result[f"benchmark.{i}.mean"]) / 1e9
                score /= num_benchmarks

                # TODO: specify what LB it saves to
                if mode == SubmissionMode.LEADERBOARD:
                    with self.bot.leaderboard_db as db:
                        db.create_submission(
                            {
                                "submission_name": script.filename,
                                "submission_time": datetime.now(),
                                "leaderboard_name": leaderboard_name,
                                "code": submission_content,
                                "user_id": interaction.user.id,
                                "submission_score": score,
                                "gpu_type": gpu.name,
                            }
                        )

                await discord_thread.send(
                    "## Result:\n"
                    + f"Leaderboard `{leaderboard_name}`:\n"
                    + f"> **{user_id}**'s `{script.filename}` on `{gpu.name}` ran "
                    + f"for `{score:.9f}` seconds!",
                )
        except Exception as e:
            logger.error("Error in leaderboard submission", exc_info=e)
            await discord_thread.send(
                "## Result:\n"
                + f"Leaderboard submission to '{leaderboard_name}' on {gpu.name} "
                + f"using {runner_name} runners failed!\n",
            )

    async def select_gpu_view(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        gpus: List[str],
    ):
        """
        UI displayed to user to select GPUs that they want to use.
        """
        view = GPUSelectionView(gpus)

        await send_discord_message(
            interaction,
            f"Please select the GPU(s) for leaderboard: {leaderboard_name}.",
            view=view,
            ephemeral=True,
        )

        await view.wait()
        return view

    async def before_submit_hook(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
    ):
        """
        Main logic to handle at the beginning of a user submission to a runner, to make
        sure reference code, deadlines, etc. are all correct.
        """
        # Read and convert reference code
        with self.bot.leaderboard_db as db:
            leaderboard_item = db.get_leaderboard(leaderboard_name)
            if not leaderboard_item:
                await send_discord_message(
                    interaction,
                    f"Leaderboard {leaderboard_name} not found.",
                    ephemeral=True,
                )
                return None, None

            now = datetime.now()
            deadline = leaderboard_item["deadline"]

            if now.date() > deadline.date():
                await send_discord_message(
                    interaction,
                    f"The deadline to submit to {leaderboard_name} has passed.\n"
                    + f"It was {deadline.date()} and today is {now.date()}.",
                )
                return None, None

            gpus = db.get_leaderboard_gpu_types(leaderboard_name)
            task = leaderboard_item["task"]
        return task, gpus

    def _get_run_command(self, gpu) -> Optional[Callable]:
        runner_cog = self.bot.get_cog(f"{gpu.runner}Cog")

        if not all([runner_cog]):
            logger.error("Cog for runner %s for gpu %s not found!", f"{gpu.runner}Cog", gpu.name)
            return None
        return runner_cog.submit_leaderboard

    @staticmethod
    def _get_popcorn_directives(submission: str) -> dict:
        popcorn_info = {"gpus": None, "leaderboard": None}
        for line in submission.splitlines():
            # only process the first comment block of the file.
            # for simplicity, don't care whether these are python or C++ comments here
            if not (line.startswith("//") or line.startswith("#")):
                break

            args = line.split()
            if args[0] in ["//!POPCORN", "#!POPCORN"]:
                arg = args[1].strip().lower()
                if arg in ["gpu", "gpus"]:
                    popcorn_info["gpus"] = args[2:]
                elif arg == "leaderboard":
                    popcorn_info["leaderboard"] = args[2]
        return popcorn_info

    async def on_submit_hook(  # noqa: C901
        self,
        interaction: discord.Interaction,
        leaderboard_name: Optional[str],
        script: discord.Attachment,
        mode: SubmissionMode,
        cmd_gpus: Optional[List[str]],
    ) -> int:
        """
        Called as the main body of a submission to route to the correct runner.
        """
        # Read the template file
        submission_content = await script.read()

        try:
            submission_content = submission_content.decode()
        except UnicodeError:
            await send_discord_message(
                interaction, "Could not decode your file. Is it UTF-8?", ephemeral=True
            )
            return -1

        info = self._get_popcorn_directives(submission_content)
        # command argument GPUs overwrites popcorn directive
        if info["gpus"] is not None and cmd_gpus is None:
            cmd_gpus = info["gpus"]

        if info["leaderboard"] is not None:
            if leaderboard_name is not None:
                await send_discord_message(
                    interaction,
                    "Contradicting leaderboard name specification. "
                    "Submitting to {leaderboard_name}",
                )
            else:
                leaderboard_name = info["leaderboard"]

        if leaderboard_name is None:
            await send_discord_message(
                interaction,
                "Missing leaderboard name. "
                "Either supply as argument in the submit command, or "
                "specify in your submission script using the "
                "`{#,//}!POPCORN leaderboard` directive.",
            )
            return -1

        task, task_gpus = await self.before_submit_hook(
            interaction,
            leaderboard_name,
        )

        # GPU selection View
        if len(task_gpus) == 0:
            await send_discord_message(
                interaction,
                "❌ No available GPUs for Leaderboard " + f"`{leaderboard_name}`.",
            )
            return -1

        # if there is more than one candidate GPU, display UI to let user select,
        # otherwise just run on that GPU
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=True)

        if cmd_gpus is not None:
            selected_gpus = []
            for g in cmd_gpus:
                if g in task_gpus:
                    selected_gpus.append(g)
                else:
                    await send_discord_message(
                        interaction,
                        f"GPU {g} not available for `{leaderboard_name}`",
                        ephemeral=True,
                    )
                    return -1
        elif len(task_gpus) == 1:
            await send_discord_message(
                interaction,
                f"Running for `{leaderboard_name}` on GPU: **{task_gpus[0]}**",
                ephemeral=True,
            )
            selected_gpus = task_gpus
        else:
            view = await self.select_gpu_view(interaction, leaderboard_name, task_gpus)
            selected_gpus = view.selected_gpus

        selected_gpus = [get_gpu_by_name(gpu) for gpu in selected_gpus]
        commands = [self._get_run_command(gpu) for gpu in selected_gpus]
        if any((c is None for c in commands)):
            await send_discord_message(interaction, "❌ Required runner not found!")
            return -1

        tasks = [
            self.async_submit_cog_job(
                interaction,
                leaderboard_name,
                script,
                command,
                task,
                submission_content,
                app_commands.Choice(name=gpu.name, value=gpu.value),
                gpu.runner,
                mode,
            )
            for gpu, command in zip(selected_gpus, commands, strict=False)
        ]

        # also schedule secret run
        if mode == SubmissionMode.LEADERBOARD:
            tasks += [
                self.async_submit_cog_job(
                    interaction,
                    leaderboard_name,
                    script,
                    command,
                    task,
                    submission_content,
                    app_commands.Choice(name=gpu.name, value=gpu.value),
                    gpu.runner,
                    SubmissionMode.PRIVATE,
                )
                for gpu, command in zip(selected_gpus, commands, strict=False)
            ]

        await asyncio.gather(*tasks)

        await send_discord_message(
            interaction,
            f"{mode.value.capitalize()} submission to '{leaderboard_name}' "
            f"on GPUS: {', '.join([gpu.name for gpu in selected_gpus])} "
            f"using {', '.join({gpu.runner for gpu in selected_gpus})} runners succeeded!",
        )
        return 0

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.channel_id != self.bot.leaderboard_submissions_id:
            await interaction.response.send_message(
                f"Please use submission commands in <#{self.bot.leaderboard_submissions_id}>",
                ephemeral=True,
            )
            return False
        return True

    async def submit(
        self,
        interaction: discord.Interaction,
        leaderboard_name: Optional[str],
        script: discord.Attachment,
        mode: SubmissionMode,
        gpu: Optional[str],
    ):
        if not self.bot.accepts_jobs:
            await send_discord_message(
                interaction,
                "The bot is currently not accepting any new submissions, please try again later.",
                ephemeral=True,
            )
            return

        if gpu is not None:
            gpu = [gpu.strip() for gpu in gpu.split(",")]
        try:
            return await self.on_submit_hook(interaction, leaderboard_name, script, mode, gpu)
        except Exception as e:
            logger.error("Error handling leaderboard submission", exc_info=e)
            # don't leak any information, but at least acknowledge that the command failed.
            await send_discord_message(
                interaction,
                f"An error occurred when submitting to leaderboard " f"`{leaderboard_name}`.",
                ephemeral=True,
            )
            return -1

    @app_commands.command(name="test", description="Start a testing/debugging run")
    @app_commands.describe(
        leaderboard_name="Name of the competition / kernel to optimize",
        script="The Python / CUDA script file to run",
        gpu="Select GPU. Leave empty for interactive or automatic selection.",
    )
    @app_commands.autocomplete(leaderboard_name=leaderboard_name_autocomplete)
    async def submit_test(
        self,
        interaction: discord.Interaction,
        script: discord.Attachment,
        leaderboard_name: Optional[str] = None,
        gpu: Optional[str] = None,
    ):
        return await self.submit(
            interaction, leaderboard_name, script, mode=SubmissionMode.TEST, gpu=gpu
        )

    @app_commands.command(name="benchmark", description="Start a benchmarking run")
    @app_commands.describe(
        leaderboard_name="Name of the competition / kernel to optimize",
        script="The Python / CUDA script file to run",
        gpu="Select GPU. Leave empty for interactive or automatic selection.",
    )
    @app_commands.autocomplete(leaderboard_name=leaderboard_name_autocomplete)
    async def submit_bench(
        self,
        interaction: discord.Interaction,
        script: discord.Attachment,
        leaderboard_name: Optional[str],
        gpu: Optional[str],
    ):
        return await self.submit(
            interaction, leaderboard_name, script, mode=SubmissionMode.BENCHMARK, gpu=gpu
        )

    @app_commands.command(
        name="ranked", description="Start a ranked run for an official leaderboard submission"
    )
    @app_commands.describe(
        leaderboard_name="Name of the competition / kernel to optimize",
        script="The Python / CUDA script file to run",
        gpu="Select GPU. Leave empty for interactive or automatic selection.",
    )
    @app_commands.autocomplete(leaderboard_name=leaderboard_name_autocomplete)
    async def submit_ranked(
        self,
        interaction: discord.Interaction,
        script: discord.Attachment,
        leaderboard_name: Optional[str] = None,
        gpu: Optional[str] = None,
    ):
        return await self.submit(
            interaction, leaderboard_name, script, mode=SubmissionMode.LEADERBOARD, gpu=gpu
        )


async def lang_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> list[discord.app_commands.Choice[str]]:
    """
    "Autocompletion" for language selection in template command.
    This does not really autocomplete; I just provides a drop-down
    with all _available_ languages for the chosen leaderboard
    (opposed to a Choice argument, which cannot adapt).
    """
    lb = interaction.namespace["leaderboard_name"]
    bot = interaction.client

    with bot.leaderboard_db as db:
        leaderboard_item = db.get_leaderboard(lb)  # type: LeaderboardItem
        if not leaderboard_item:
            raise ValueError("Invalid leaderboard")

    candidates = leaderboard_item["task"].templates
    return [discord.app_commands.Choice(name=c, value=c) for c in candidates]


def add_header_to_template(lang: str, lb: LeaderboardItem):
    template_file = lb["task"].templates[lang]

    comment_char = {"CUDA": "//", "Python": "#", "Triton": "#"}[lang]

    description_comment = [
        f"{comment_char} > {line}\n" for line in lb["task"].description.splitlines()
    ]
    header = f"""
{comment_char}!POPCORN leaderboard {lb['name']}

{comment_char} This is a submission template for popcorn leaderboard '{lb['name']}'.
{comment_char} Your task is as follows:
{str.join('\n', description_comment)}
{comment_char} The deadline for this leaderboard is {lb["deadline"]}

{comment_char} You can automatically route this file to specific GPUs by adding a line
{comment_char} `{comment_char}!POPCORN gpus <GPUs>` to the header of this file.
{comment_char} Happy hacking!

"""[1:]
    return header + template_file + "\n"


class LeaderboardCog(commands.Cog):
    def __init__(self, bot: "ClusterBot"):
        self.bot = bot

        bot.leaderboard_group.add_command(LeaderboardSubmitCog(bot))

        self.get_leaderboards = bot.leaderboard_group.command(
            name="list", description="Get all leaderboards"
        )(self.get_leaderboards)

        self.get_leaderboard_submissions = bot.leaderboard_group.command(
            name="show", description="Get all submissions for a leaderboard"
        )(self.get_leaderboard_submissions)

        self.get_user_leaderboard_submissions = bot.leaderboard_group.command(
            name="show-personal", description="Get all your submissions for a leaderboard"
        )(self.get_user_leaderboard_submissions)

        self.get_leaderboard_task = bot.leaderboard_group.command(
            name="task", description="Get leaderboard reference codes"
        )(self.get_leaderboard_task)

        self.get_task_template = bot.leaderboard_group.command(
            name="template", description="Get a starter template file for a task"
        )(self.get_task_template)

        # Start updating leaderboard
        self.leaderboard_update.start()

    # --------------------------------------------------------------------------
    # |                           LOOPING FUNCTIONS                            |
    # --------------------------------------------------------------------------
    @tasks.loop(minutes=1)
    async def leaderboard_update(self):
        """Task that updates the leaderboard every minute."""
        for guild in self.bot.guilds:
            channel = await self.ensure_channel_exists(guild, "active-leaderboards")

            # Get the pinned message or create a new one
            pinned_messages = await channel.pins()
            if pinned_messages:
                message = pinned_messages[0]
            else:
                message = await channel.send("Loading leaderboard...")
                await message.pin()

            # Update the leaderboard message
            embed, view = await self._get_leaderboard_helper()

            if embed:
                await message.edit(content="", embed=embed, view=view)
            else:
                await message.edit(content="No active leaderboards.")

    @leaderboard_update.before_loop
    async def before_leaderboard_update(self):
        """Wait for the bot to be ready before starting the task."""
        await self.bot.wait_until_ready()

    # --------------------------------------------------------------------------
    # |                           HELPER FUNCTIONS                              |
    # --------------------------------------------------------------------------
    async def ensure_channel_exists(self, guild, channel_name):
        """Ensure the leaderboard channel exists, and create it if not."""
        channel = discord.utils.get(guild.text_channels, name=channel_name)
        if not channel:
            channel = await guild.create_text_channel(channel_name)
        return channel

    async def _display_lb_submissions_helper(
        self,
        submissions,
        interaction,
        leaderboard_name: str,
        gpu: str,
        user_id: Optional[int] = None,
    ):
        """
        Display leaderboard submissions for a particular GPU to discord.
        Must be used as a follow-up currently.
        """

        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=True)

        if not submissions:
            await send_discord_message(
                interaction,
                f'No submissions found for "{leaderboard_name}".',
                ephemeral=True,
            )
            return

        # Create embed
        processed_submissions = [
            {
                "Rank": submission["rank"],
                "User": await get_user_from_id(submission["user_id"], interaction, self.bot),
                "Score": f"{submission['submission_score']:.9f}",
                "Submission Name": submission["submission_name"],
            }
            for submission in submissions
        ]

        title = f'Leaderboard Submissions for "{leaderboard_name}" on {gpu}'
        if user_id:
            title += f" for user {await get_user_from_id(user_id, interaction, self.bot)}"

        column_widths = {
            "Rank": 4,
            "User": 14,
            "Score": 12,
            "Submission Name": 14,
        }
        embed, view = create_table(
            title,
            processed_submissions,
            items_per_page=5,
            column_widths=column_widths,
        )

        await send_discord_message(
            interaction,
            "",
            embed=embed,
            view=view,
            ephemeral=True,
        )

    async def _get_submissions_helper(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        user_id: str = None,
    ):
        """Helper method to get leaderboard submissions with optional user filtering"""
        try:
            submissions = {}
            with self.bot.leaderboard_db as db:
                leaderboard_id = db.get_leaderboard(leaderboard_name)["id"]
                if not leaderboard_id:
                    await send_discord_message(
                        interaction,
                        f'Leaderboard "{leaderboard_name}" not found.',
                        ephemeral=True,
                    )
                    return

                gpus = db.get_leaderboard_gpu_types(leaderboard_name)
                for gpu in gpus:
                    submissions[gpu] = db.get_leaderboard_submissions(
                        leaderboard_name, gpu, user_id
                    )

            if not interaction.response.is_done():
                await interaction.response.defer(ephemeral=True)

            view = GPUSelectionView(gpus)
            await send_discord_message(
                interaction,
                f"Please select GPUs view for leaderboard: {leaderboard_name}.",
                view=view,
                ephemeral=True,
            )

            await view.wait()

            for gpu in view.selected_gpus:
                await self._display_lb_submissions_helper(
                    submissions[gpu],
                    interaction,
                    leaderboard_name,
                    gpu,
                    user_id,
                )

        except Exception as e:
            logger.error(str(e), exc_info=e)
            if "'NoneType' object is not subscriptable" in str(e):
                await send_discord_message(
                    interaction,
                    f"The leaderboard '{leaderboard_name}' doesn't exist.",
                    ephemeral=True,
                )
            else:
                await send_discord_message(
                    interaction, "An unknown error occurred.", ephemeral=True
                )

    async def _get_leaderboard_helper(self):
        """
        Helper for grabbing the leaderboard DB and forming the
        renderable item.
        """
        with self.bot.leaderboard_db as db:
            leaderboards = db.get_leaderboards()

        if not leaderboards:
            return None, None

        to_show = [
            {
                "Name": x["name"],
                "Deadline": x["deadline"].strftime("%Y-%m-%d %H:%M"),
                "GPU Types": ", ".join(x["gpu_types"]),
            }
            for x in leaderboards
        ]

        column_widths = {
            "Name": 18,
            "Deadline": 18,
            "GPU Types": 11,
        }
        embed, view = create_table(
            "Active Leaderboards",
            to_show,
            items_per_page=5,
            column_widths=column_widths,
        )

        return embed, view

    # --------------------------------------------------------------------------
    # |                           COMMANDS                                      |
    # --------------------------------------------------------------------------

    async def get_leaderboards(self, interaction: discord.Interaction):
        """Display all leaderboards in a table format"""
        await interaction.response.defer(ephemeral=True)

        embed, view = await self._get_leaderboard_helper()

        if not embed:
            await send_discord_message(interaction, "No leaderboards found.", ephemeral=True)
            return

        await send_discord_message(
            interaction,
            "",
            embed=embed,
            view=view,
            ephemeral=True,
        )

    @app_commands.describe(leaderboard_name="Name of the leaderboard")
    @app_commands.autocomplete(leaderboard_name=leaderboard_name_autocomplete)
    async def get_leaderboard_task(self, interaction: discord.Interaction, leaderboard_name: str):
        await interaction.response.defer(ephemeral=True)

        with self.bot.leaderboard_db as db:
            leaderboard_item = db.get_leaderboard(leaderboard_name)  # type: LeaderboardItem
            if not leaderboard_item:
                await send_discord_message(interaction, "Leaderboard not found.", ephemeral=True)
                return

        code = leaderboard_item["task"].files

        files = []
        for file_name, content in code.items():
            files.append(discord.File(fp=StringIO(content), filename=file_name))

        message = f"**Reference Code for {leaderboard_name}**\n"

        await send_discord_message(interaction, message, ephemeral=True, files=files)

    @app_commands.describe(
        leaderboard_name="Name of the leaderboard",
        lang="The programming language for which to download a template file.",
    )
    @app_commands.autocomplete(
        leaderboard_name=leaderboard_name_autocomplete, lang=lang_autocomplete
    )
    async def get_task_template(
        self, interaction: discord.Interaction, leaderboard_name: str, lang: str
    ):
        await interaction.response.defer(ephemeral=True)

        try:
            with self.bot.leaderboard_db as db:
                leaderboard_item = db.get_leaderboard(leaderboard_name)  # type: LeaderboardItem
                if not leaderboard_item:
                    await send_discord_message(
                        interaction, "Leaderboard not found.", ephemeral=True
                    )
                    return

            if lang not in leaderboard_item["task"].templates:
                langs = "\n".join(
                    (f"* {lang} " for lang in leaderboard_item["task"].templates.keys())
                )
                await send_discord_message(
                    interaction,
                    f"Task `{leaderboard_name}` does not have a template for `{lang}`.\n"
                    f"Choose from:\n{langs}",
                    ephemeral=True,
                )
                return

            template = add_header_to_template(lang, leaderboard_item)
            ext = {"CUDA": "cu", "Python": "py", "Triton": "py"}
            file_name = f"{leaderboard_name}.{ext[lang]}"
            file = discord.File(fp=StringIO(template), filename=file_name)
            message = f"**Starter code for {leaderboard_name}**\n"
            await send_discord_message(interaction, message, ephemeral=True, file=file)
        except Exception as E:
            logger.exception(
                "Error fetching template %s for %s", lang, leaderboard_name, exc_info=E
            )
            await send_discord_message(
                interaction,
                f"Could not fetch template {lang} for {leaderboard_name}",
                ephemeral=True,
            )
            return

    @discord.app_commands.describe(leaderboard_name="Name of the leaderboard")
    @app_commands.autocomplete(leaderboard_name=leaderboard_name_autocomplete)
    async def get_leaderboard_submissions(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
    ):
        await self._get_submissions_helper(interaction, leaderboard_name)

    @discord.app_commands.describe(leaderboard_name="Name of the leaderboard")
    @app_commands.autocomplete(leaderboard_name=leaderboard_name_autocomplete)
    async def get_user_leaderboard_submissions(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
    ):
        await self._get_submissions_helper(interaction, leaderboard_name, str(interaction.user.id))
