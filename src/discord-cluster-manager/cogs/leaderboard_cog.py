import asyncio
from datetime import datetime, timedelta
from io import StringIO
from typing import TYPE_CHECKING, List, Optional

import discord
from consts import (
    SubmissionMode,
    get_gpu_by_name,
)
from discord import app_commands
from discord.ext import commands, tasks
from leaderboard_db import leaderboard_name_autocomplete
from report import MultiProgressReporter
from submission import SubmissionRequest, prepare_submission
from ui.misc import GPUSelectionView
from ui.table import create_table
from utils import (
    LeaderboardItem,
    LeaderboardRankedEntry,
    RunItem,
    SubmissionItem,
    format_time,
    get_user_from_id,
    send_discord_message,
    setup_logging,
    with_error_handling,
)

if TYPE_CHECKING:
    from ..bot import ClusterBot

logger = setup_logging()


class LeaderboardSubmitCog(app_commands.Group):
    def __init__(self, bot: "ClusterBot"):
        super().__init__(name="submit", description="Submit to leaderboard")
        self.bot = bot

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

        req = SubmissionRequest(
            code=submission_content,
            file_name=script.filename,
            user_id=interaction.user.id,
            gpus=cmd_gpus,
            leaderboard=leaderboard_name,
        )
        req = prepare_submission(req, self.bot.leaderboard_db)

        # if there is more than one candidate GPU, display UI to let user select,
        # otherwise just run on that GPU
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=True)

        if req.gpus is None:
            view = await self.select_gpu_view(interaction, leaderboard_name, req.task_gpus)
            selected_gpus = view.selected_gpus
        else:
            selected_gpus = req.gpus

        selected_gpus = [get_gpu_by_name(gpu) for gpu in selected_gpus]

        command = self.bot.get_cog("SubmitCog").submit_leaderboard

        user_name = interaction.user.global_name or interaction.user.name
        # Create a submission entry in the database
        with self.bot.leaderboard_db as db:
            sub_id = db.create_submission(
                leaderboard=req.leaderboard,
                file_name=script.filename,
                code=submission_content,
                user_id=interaction.user.id,
                time=datetime.now(),
                user_name=user_name,
            )

        run_msg = f"Submission **{sub_id}**: `{script.filename}` for `{leaderboard_name}`"
        reporter = MultiProgressReporter(run_msg)
        try:
            tasks = [
                command(
                    interaction,
                    sub_id,
                    submission_content,
                    script.filename,
                    gpu,
                    reporter.add_run(f"{gpu.name} on {gpu.runner}"),
                    req.task,
                    mode,
                    None,
                )
                for gpu in selected_gpus
            ]

            # also schedule secret run
            if mode == SubmissionMode.LEADERBOARD:
                tasks += [
                    command(
                        interaction,
                        sub_id,
                        script,
                        gpu,
                        reporter.add_run(f"{gpu.name} on {gpu.runner} (secret)"),
                        req.task,
                        SubmissionMode.PRIVATE,
                        req.secret_seed,
                    )
                    for gpu in selected_gpus
                ]
            await reporter.show(interaction)
            await asyncio.gather(*tasks)
        finally:
            with self.bot.leaderboard_db as db:
                db.mark_submission_done(sub_id)

        if mode == SubmissionMode.LEADERBOARD:
            await self.post_submit_hook(interaction, sub_id)
        return sub_id

    def generate_run_verdict(self, run: RunItem, sub_data: SubmissionItem):
        medals = {1: "🥇 First", 2: "🥈 Second", 3: "🥉 Third"}

        # get the competition
        with self.bot.leaderboard_db as db:
            competition = db.get_leaderboard_submissions(
                sub_data["leaderboard_name"], run["runner"]
            )
        # compare against the competition
        other_by_user = False
        run_time = float(run["score"])
        score_text = format_time(run_time * 1e9)

        for entry in competition:
            # can we find our own run? Only if it is the fastest submission by this user
            if entry["submission_id"] == sub_data["submission_id"]:
                rank = entry["rank"]
                if 1 <= rank <= 3:
                    return f"> {medals[rank]} place on {run['runner']}: {score_text}"
                elif rank <= 10:
                    return f"> {rank}th place on {run['runner']}: {score_text}"
                else:
                    return f"> Personal best on {run['runner']}: {score_text}"
            elif entry["user_id"] == sub_data["user_id"]:
                other_by_user = True
        if other_by_user:
            # User already has a submission that is faster
            return f"> Successful on {run['runner']}: {score_text}"
        else:
            # no submission by the user exists
            return f"> 🍾 First successful submission on {run['runner']}: {score_text}"

    async def post_submit_hook(self, interaction: discord.Interaction, sub_id: int):
        with self.bot.leaderboard_db as db:
            sub_data: SubmissionItem = db.get_submission_by_id(sub_id)

        result_lines = []
        for run in sub_data["runs"]:
            if (
                not run["secret"]
                and run["mode"] == SubmissionMode.LEADERBOARD.value
                and run["passed"]
            ):
                result_lines.append(self.generate_run_verdict(run, sub_data))

        if len(result_lines) > 0:
            await send_discord_message(
                interaction,
                f"{interaction.user.mention}'s submission with id `{sub_id}` to leaderboard `{sub_data['leaderboard_name']}`:\n"  # noqa: E501
                + "\n".join(result_lines),
            )

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.channel_id != self.bot.leaderboard_submissions_id:
            await interaction.response.send_message(
                f"Submissions are only allowed in <#{self.bot.leaderboard_submissions_id}> channel",
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
                f"An error occurred when submitting to leaderboard `{leaderboard_name}`.",
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
    @with_error_handling
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
    @with_error_handling
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
    @with_error_handling
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
{comment_char}!POPCORN leaderboard {lb["name"]}

{comment_char} This is a submission template for popcorn leaderboard '{lb["name"]}'.
{comment_char} Your task is as follows:
{str.join("\n", description_comment)}
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

        self.get_submission_by_id = bot.leaderboard_group.command(
            name="get-submission", description="Retrieve one of your past submissions"
        )(self.get_submission_by_id)

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
                await message.edit(content="There are currently no active leaderboards.")

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
        submissions: list[LeaderboardRankedEntry],
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
                f"There are currently no submissions for leaderboard `{leaderboard_name}`.",
                ephemeral=True,
            )
            return

        # Create embed
        if user_id is None:
            processed_submissions = [
                {
                    "Rank": submission["rank"],
                    "User": await get_user_from_id(submission["user_id"], interaction, self.bot),
                    "Score": f"{format_time(float(submission['submission_score']) * 1e9)}",
                    "Submission Name": submission["submission_name"],
                }
                for submission in submissions
            ]
            column_widths = {
                "Rank": 4,
                "User": 14,
                "Score": 12,
                "Submission Name": 14,
            }
        else:

            def _time(t: datetime):
                if (datetime.now(tz=t.tzinfo) - t) > timedelta(hours=24):
                    return t.strftime("%y-%m-%d")
                else:
                    return t.strftime("%H:%M:%S")

            processed_submissions = [
                {
                    "Rank": submission["rank"],
                    "ID": submission["submission_id"],
                    "Score": f"{format_time(float(submission['submission_score']) * 1e9)}",
                    "Submission Name": submission["submission_name"],
                    "Time": _time(submission["submission_time"]),
                }
                for submission in submissions
            ]
            column_widths = {
                "ID": 5,
                "Rank": 4,
                "Score": 10,
                "Submission Name": 14,
                "Time": 8,
            }

        title = f'Leaderboard Submissions for "{leaderboard_name}" on {gpu}'
        if user_id:
            title += f" for user {await get_user_from_id(user_id, interaction, self.bot)}"

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
                f"Please select GPUs to view for leaderboard `{leaderboard_name}`. ",
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

    @with_error_handling
    async def get_leaderboards(self, interaction: discord.Interaction):
        """Display all leaderboards in a table format"""
        await interaction.response.defer(ephemeral=True)

        embed, view = await self._get_leaderboard_helper()

        if not embed:
            await send_discord_message(
                interaction, "There are currently no active leaderboards.", ephemeral=True
            )
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
    @with_error_handling
    async def get_leaderboard_task(self, interaction: discord.Interaction, leaderboard_name: str):
        await interaction.response.defer(ephemeral=True)

        with self.bot.leaderboard_db as db:
            leaderboard_item = db.get_leaderboard(leaderboard_name)  # type: LeaderboardItem
            if not leaderboard_item:
                await send_discord_message(
                    interaction,
                    f"Leaderboard with name `{leaderboard_name}` not found.",
                    ephemeral=True,
                )
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
    @with_error_handling
    async def get_task_template(
        self, interaction: discord.Interaction, leaderboard_name: str, lang: str
    ):
        await interaction.response.defer(ephemeral=True)

        try:
            with self.bot.leaderboard_db as db:
                leaderboard_item = db.get_leaderboard(leaderboard_name)  # type: LeaderboardItem
                if not leaderboard_item:
                    await send_discord_message(
                        interaction,
                        f"Leaderboard with name `{leaderboard_name}` not found.",
                        ephemeral=True,
                    )
                    return

            if lang not in leaderboard_item["task"].templates:
                langs = "\n".join(
                    (f"* {lang} " for lang in leaderboard_item["task"].templates.keys())
                )
                await send_discord_message(
                    interaction,
                    f"Leaderboard `{leaderboard_name}` does not have a template for `{lang}`.\n"  # noqa: E501
                    f"Choose one of:\n{langs}",
                    ephemeral=True,
                )
                return

            template = add_header_to_template(lang, leaderboard_item)
            ext = {"CUDA": "cu", "Python": "py", "Triton": "py"}
            file_name = f"{leaderboard_name}.{ext[lang]}"
            file = discord.File(fp=StringIO(template), filename=file_name)
            message = f"**Starter code for leaderboard `{leaderboard_name}`**\n"
            await send_discord_message(interaction, message, ephemeral=True, file=file)
        except Exception as E:
            logger.exception(
                "Error fetching template %s for %s", lang, leaderboard_name, exc_info=E
            )
            await send_discord_message(
                interaction,
                f"Could not find a template with language `{lang}` for leaderboard `{leaderboard_name}`",  # noqa: E501
                ephemeral=True,
            )
            return

    @discord.app_commands.describe(leaderboard_name="Name of the leaderboard")
    @app_commands.autocomplete(leaderboard_name=leaderboard_name_autocomplete)
    @with_error_handling
    async def get_leaderboard_submissions(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
    ):
        await self._get_submissions_helper(interaction, leaderboard_name)

    @discord.app_commands.describe(leaderboard_name="Name of the leaderboard")
    @app_commands.autocomplete(leaderboard_name=leaderboard_name_autocomplete)
    @with_error_handling
    async def get_user_leaderboard_submissions(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
    ):
        await self._get_submissions_helper(interaction, leaderboard_name, str(interaction.user.id))

    @discord.app_commands.describe(submission_id="ID of the submission")
    @with_error_handling
    async def get_submission_by_id(
        self,
        interaction: discord.Interaction,
        submission_id: int,
    ):
        with self.bot.leaderboard_db as db:
            sub: SubmissionItem = db.get_submission_by_id(submission_id)

        # allowed/possible to see submission
        if sub is None or int(sub["user_id"]) != interaction.user.id:
            await send_discord_message(
                interaction,
                f"Submission with id `{submission_id}` is not one of your submissions",
                ephemeral=True,
            )
            return

        msg = f"# Submission {submission_id}\n"
        msg += f"submitted on {sub['submission_time']}"
        msg += f" to leaderboard `{sub['leaderboard_name']}`."
        if not sub["done"]:
            msg += "\n*Submission is still running!*\n"

        file = discord.File(fp=StringIO(sub["code"]), filename=sub["file_name"])

        if len(sub["runs"]) > 0:
            msg += "\nRuns:\n"
        for run in sub["runs"]:
            if run["secret"]:
                continue

            msg += f" * {run['mode']} on {run['runner']}: "
            if run["score"] is not None and run["passed"]:
                msg += f"{run['score']}"
            else:
                msg += "pass" if run["passed"] else "fail"
            msg += "\n"

        await send_discord_message(interaction, msg, ephemeral=True, file=file)
