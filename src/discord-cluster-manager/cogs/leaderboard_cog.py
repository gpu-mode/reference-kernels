import asyncio
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Callable, List, Optional, Type

import discord
from consts import (
    GPU_SELECTION,
    AllGPU,
    GitHubGPU,
    ModalGPU,
    SubmissionMode,
)
from discord import app_commands
from discord.ext import commands, tasks
from leaderboard_db import leaderboard_name_autocomplete
from task import LeaderboardTask, make_task
from ui.misc import DeleteConfirmationModal, GPUSelectionView
from ui.table import create_table
from utils import (
    get_user_from_id,
    send_discord_message,
    setup_logging,
)

logger = setup_logging()


async def leaderboard_dir_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> list[discord.app_commands.Choice[str]]:
    """Return leaderboard names that match the current typed name"""
    root = Path("examples")
    return [
        discord.app_commands.Choice(name=x.name, value=x.name) for x in root.iterdir() if x.is_dir()
    ]


class LeaderboardSubmitCog(app_commands.Group):
    def __init__(self, bot):
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
        gpu: AllGPU,
        runner_name: str,
        mode: SubmissionMode,
    ):
        discord_thread, result = await command(
            interaction,
            script,
            app_commands.Choice(
                name=gpu.name,
                value=gpu.value,
            ),
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

                if result.runs["test"].result["check"] != "pass":
                    await discord_thread.send(
                        f"Ran on {gpu.name} using {runner_name} runners!\n"
                        + f"Leaderboard '{leaderboard_name}'.\n"
                        + f"Submission title: {script.filename}.\n"
                        + f"Submission user: {user_id}.\n"
                    )
                    return

                # TODO: Make this more flexible, not just functional
                score = 0.0
                num_benchmarks = int(result.runs["benchmark"].result["benchmark-count"])
                for i in range(num_benchmarks):
                    score += float(result.runs["benchmark"].result[f"benchmark.{i}.mean"]) / 1e9
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
        script: discord.Attachment,
    ):
        """
        Main logic to handle at the beginning of a user submission to a runner, to make
        sure reference code, deadlines, etc. are all correct.
        """
        # Read and convert reference code
        with self.bot.leaderboard_db as db:
            leaderboard_item = db.get_leaderboard(leaderboard_name)

            now = datetime.now()
            deadline = leaderboard_item["deadline"]

            if now.date() > deadline.date():
                await send_discord_message(
                    interaction,
                    f"The deadline to submit to {leaderboard_name} has passed.\n"
                    + f"It was {deadline.date()} and today is {now.date()}.",
                )
                return None

            if not leaderboard_item:
                await send_discord_message(
                    interaction,
                    f"Leaderboard {leaderboard_name} not found.",
                    ephemeral=True,
                )
                return None

            gpus = db.get_leaderboard_gpu_types(leaderboard_name)
            task = leaderboard_item["task"]

        # Read the template file
        submission_content = await script.read()

        try:
            submission_content = submission_content.decode()
        except UnicodeError:
            await send_discord_message(
                interaction, "Could not decode your file. Is it UTF-8?", ephemeral=True
            )
            return None

        return submission_content, task, gpus

    async def on_submit_hook(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        script: discord.Attachment,
        command: Callable,
        GPUsEnum: Type[Enum],
        runner_name: str,
        mode: SubmissionMode,
    ) -> int:
        """
        Called as the main body of a submission to route to the correct runner.
        """
        submission_content, task, gpus = await self.before_submit_hook(
            interaction,
            leaderboard_name,
            script,
        )

        # GPU selection View
        gpu_enums = {e.name for e in GPUsEnum}
        gpus = [gpu for gpu in gpus if gpu in gpu_enums]

        if len(gpus) == 0:
            await send_discord_message(
                interaction,
                "❌ No available GPUs for Leaderboard "
                + f"`{leaderboard_name}` on {runner_name} runner.",
            )
            return -1

        # if there is more than one candidate GPU, display UI to let user select,
        # otherwise just run on that GPU
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=True)
        if len(gpus) == 1:
            await send_discord_message(
                interaction,
                f"Running for `{leaderboard_name}` on GPU: **{gpus[0]}**",
                ephemeral=True,
            )
            selected_gpus = gpus
        else:
            view = await self.select_gpu_view(interaction, leaderboard_name, gpus)
            selected_gpus = view.selected_gpus

        tasks = [
            self.async_submit_cog_job(
                interaction,
                leaderboard_name,
                script,
                command,
                task,
                submission_content,
                AllGPU[gpu],
                runner_name,
                mode,
            )
            for gpu in selected_gpus
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
                    AllGPU[gpu],
                    runner_name,
                    SubmissionMode.PRIVATE,
                )
                for gpu in selected_gpus
            ]

        await asyncio.gather(*tasks)
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
        runner_name: str,
        interaction: discord.Interaction,
        leaderboard_name: str,
        script: discord.Attachment,
        mode: SubmissionMode,
    ):
        # Call Modal runner
        runner_cog = self.bot.get_cog(f"{runner_name}Cog")

        if not all([runner_cog]):
            await send_discord_message(interaction, f"❌ Required {runner_name} cogs not found!")
            return

        runner_command = runner_cog.submit_leaderboard

        try:
            return await self.on_submit_hook(
                interaction,
                leaderboard_name,
                script,
                runner_command,
                GPU_SELECTION[runner_name],
                runner_name,
                mode,
            )
        except Exception as e:
            logger.error("Error handling leaderboard submission", exc_info=e)
            # don't leak any information, but at least acknowledge that the command failed.
            await send_discord_message(
                interaction,
                f"An error occurred when submitting to leaderboard "
                f"`{leaderboard_name}` on runner `{runner_name}`.",
                ephemeral=True,
            )
            return -1

    @app_commands.command(name="modal", description="Submit leaderboard data for modal")
    @app_commands.describe(
        leaderboard_name="Name of the competition / kernel to optimize",
        script="The Python / CUDA script file to run",
    )
    @app_commands.autocomplete(leaderboard_name=leaderboard_name_autocomplete)
    async def submit_modal(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        script: discord.Attachment,
    ):
        return await self.submit(
            "Modal", interaction, leaderboard_name, script, mode=SubmissionMode.LEADERBOARD
        )

    @app_commands.command(name="github", description="Submit leaderboard data for GitHub")
    @app_commands.describe(
        leaderboard_name="Name of the competition / kernel to optimize",
        script="The Python / CUDA script file to run",
    )
    @app_commands.autocomplete(leaderboard_name=leaderboard_name_autocomplete)
    async def submit_github(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        script: discord.Attachment,
    ):
        return await self.submit(
            "GitHub", interaction, leaderboard_name, script, mode=SubmissionMode.LEADERBOARD
        )

    @app_commands.command(name="test", description="Start a testing/debugging run")
    @app_commands.describe(
        leaderboard_name="Name of the competition / kernel to optimize",
        runner="Name of the runner to run on",
        script="The Python / CUDA script file to run",
    )
    @app_commands.autocomplete(leaderboard_name=leaderboard_name_autocomplete)
    async def submit_test(
        self,
        interaction: discord.Interaction,
        runner: str,
        leaderboard_name: str,
        script: discord.Attachment,
    ):
        runner = {"github": "GitHub", "modal": "Modal"}[runner.lower()]
        return await self.submit(
            runner, interaction, leaderboard_name, script, mode=SubmissionMode.TEST
        )

    @app_commands.command(name="benchmark", description="Start a benchmarking run")
    @app_commands.describe(
        leaderboard_name="Name of the competition / kernel to optimize",
        runner="Name of the runner to run on",
        script="The Python / CUDA script file to run",
    )
    @app_commands.autocomplete(leaderboard_name=leaderboard_name_autocomplete)
    async def submit_bench(
        self,
        interaction: discord.Interaction,
        runner: str,
        leaderboard_name: str,
        script: discord.Attachment,
    ):
        runner = {"github": "GitHub", "modal": "Modal"}[runner.lower()]
        return await self.submit(
            runner, interaction, leaderboard_name, script, mode=SubmissionMode.BENCHMARK
        )


class LeaderboardCog(commands.Cog):
    def __init__(self, bot):
        self.bot: commands.Bot = bot

        bot.leaderboard_group.add_command(LeaderboardSubmitCog(bot))

        self.get_leaderboards = bot.leaderboard_group.command(
            name="list", description="Get all leaderboards"
        )(self.get_leaderboards)

        self.leaderboard_create = bot.leaderboard_group.command(
            name="create", description="Create a new leaderboard"
        )(self.leaderboard_create)

        self.leaderboard_create_local = bot.leaderboard_group.command(
            name="create-local", description="Create or update a leaderboard from a local directory"
        )(self.leaderboard_create_local)

        self.delete_leaderboard = bot.leaderboard_group.command(
            name="delete", description="Delete a leaderboard"
        )(self.delete_leaderboard)

        self.get_leaderboard_submissions = bot.leaderboard_group.command(
            name="show", description="Get all submissions for a leaderboard"
        )(self.get_leaderboard_submissions)

        self.get_user_leaderboard_submissions = bot.leaderboard_group.command(
            name="show-personal", description="Get all your submissions for a leaderboard"
        )(self.get_user_leaderboard_submissions)

        self.get_leaderboard_task = bot.leaderboard_group.command(
            name="task", description="Get leaderboard reference codes"
        )(self.get_leaderboard_task)

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

    async def admin_check(self, interaction: discord.Interaction) -> bool:
        if not interaction.user.get_role(self.bot.leaderboard_admin_role_id):
            return False
        return True

    async def creator_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.get_role(self.bot.leaderboard_creator_role_id):
            return True
        return False

    async def is_creator_check(
        self, interaction: discord.Interaction, leaderboard_name: str
    ) -> bool:
        with self.bot.leaderboard_db as db:
            leaderboard_item = db.get_leaderboard(leaderboard_name)
            if leaderboard_item["creator_id"] == interaction.user.id:
                return True
            return False

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

    @discord.app_commands.describe(
        directory="Directory of the kernel definition. Also used as the leaderboard's name",
        gpu="The GPU to submit to. Leave empty for interactive selection/multiple GPUs",
    )
    @app_commands.autocomplete(directory=leaderboard_dir_autocomplete)
    @app_commands.choices(
        gpu=[app_commands.Choice(name=gpu.name, value=gpu.value) for gpu in GitHubGPU]
        + [app_commands.Choice(name=gpu.name, value=gpu.value) for gpu in ModalGPU]
    )
    async def leaderboard_create_local(
        self,
        interaction: discord.Interaction,
        directory: str,
        gpu: Optional[app_commands.Choice[str]],
    ):
        is_admin = await self.admin_check(interaction)
        if not is_admin:
            await send_discord_message(
                interaction,
                "Debug command, only for admins.",
                ephemeral=True,
            )
            return

        directory = Path("examples") / directory
        assert directory.resolve().is_relative_to(Path.cwd())
        task = make_task(directory)

        # clearly mark this leaderboard as development-only
        leaderboard_name = directory.name + "-dev"

        # create-local overwrites existing leaderboard
        with self.bot.leaderboard_db as db:
            db.delete_leaderboard(leaderboard_name)

        if await self.create_leaderboard_in_db(
            interaction,
            leaderboard_name,
            datetime.now(timezone.utc) + timedelta(days=365),
            task=task,
            gpu=gpu.value if gpu else None,
        ):
            await send_discord_message(
                interaction,
                f"Leaderboard '{leaderboard_name}' created.",
            )

    @discord.app_commands.describe(
        leaderboard_name="Name of the leaderboard",
        deadline="Competition deadline in the form: 'Y-m-d'",
        task_zip="Zipfile containing the task",
    )
    async def leaderboard_create(  # noqa: C901
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        deadline: str,
        task_zip: discord.Attachment,
    ):
        is_admin = await self.admin_check(interaction)
        is_creator = await self.creator_check(interaction)
        thread = None
        if len(leaderboard_name) > 95:
            await send_discord_message(
                interaction,
                "Leaderboard name is too long. Please keep it under 95 characters.",
                ephemeral=True,
            )
            return

        if not (is_admin or is_creator):
            await send_discord_message(
                interaction,
                "You need the Leaderboard Creator role or the Leaderboard Admin role to use this command.",  # noqa: E501
                ephemeral=True,
            )
            return

        # Try parsing with time first
        try:
            date_value = datetime.strptime(deadline, "%Y-%m-%d %H:%M")
        except ValueError:
            try:
                date_value = datetime.strptime(deadline, "%Y-%m-%d")
            except ValueError as ve:
                logger.error(f"Value Error: {str(ve)}", exc_info=True)
                await send_discord_message(
                    interaction,
                    "Invalid date format. Please use YYYY-MM-DD or YYYY-MM-DD HH:MM",
                    ephemeral=True,
                )
                return

        if date_value < datetime.now():
            await send_discord_message(
                interaction,
                f"Deadline {date_value} has already passed.",
                ephemeral=True,
            )
            return

        try:
            # Read the template file
            with tempfile.TemporaryDirectory() as tmpdir:
                with tempfile.NamedTemporaryFile("w+b") as temp:
                    temp.write(await task_zip.read())
                    temp.flush()
                    with zipfile.ZipFile(temp, "r") as zip_ref:
                        zip_ref.extractall(tmpdir)

                contents = list(Path(tmpdir).iterdir())
                # support both a zipped directory, and files
                # directly in the zip
                if len(contents) == 1:
                    task = make_task(contents[0])
                else:
                    task = make_task(tmpdir)

            success = await self.create_leaderboard_in_db(
                interaction, leaderboard_name, date_value, task
            )
            if not success:
                return

            forum_channel = self.bot.get_channel(self.bot.leaderboard_forum_id)

            existing_threads = [
                thread for thread in forum_channel.threads if thread.name == leaderboard_name
            ]

            if not existing_threads:
                thread = await forum_channel.create_thread(
                    name=leaderboard_name,
                    content=(
                        f"# New Leaderboard: {leaderboard_name}\n\n"
                        f"**Deadline**: {date_value.strftime('%Y-%m-%d %H:%M')}\n\n"
                        f"{task.description}\n\n"
                        "Submit your entries using `/submit github` or `/submit modal` in the submissions channel.\n\n"  # noqa: E501
                        f"Good luck to all participants! 🚀 <@&{self.bot.leaderboard_participant_role_id}>"  # noqa: E501
                    ),
                    auto_archive_duration=10080,  # 7 days
                )

                await send_discord_message(
                    interaction,
                    f"Leaderboard '{leaderboard_name}'.\n"
                    + f"Submission deadline: {date_value}"
                    + f"\nForum thread: {thread.thread.mention}",
                )
                return

        except discord.Forbidden:
            await send_discord_message(
                interaction,
                "Error: Bot doesn't have permission to create forum threads. Leaderboard was not created.",  # noqa: E501
                ephemeral=True,
            )
        except discord.HTTPException:
            await send_discord_message(
                interaction,
                "Error creating forum thread. Leaderboard was not created.",
                ephemeral=True,
            )
        except FileNotFoundError as e:
            file = Path(e.filename).name
            if file == "task.yml":
                await send_discord_message(
                    interaction,
                    "Error in leaderboard creation. Missing `task.yml`.",
                    ephemeral=True,
                )
            else:
                await send_discord_message(
                    interaction,
                    f"Error in leaderboard creation. Could not find `{file}`.",
                    ephemeral=True,
                )
        except zipfile.BadZipFile:
            # Handle any other errors
            await send_discord_message(
                interaction,
                "Error in leaderboard creation. Is the uploaded file a valid zip archive?",
                ephemeral=True,
            )
        except Exception as e:
            logger.error(f"Error in leaderboard creation: {e}", exc_info=e)
            # Handle any other errors
            await send_discord_message(
                interaction,
                "Error in leaderboard creation.",
                ephemeral=True,
            )
        if thread:
            await thread.thread.delete()

        with self.bot.leaderboard_db as db:  # Cleanup in case lb was created
            db.delete_leaderboard(leaderboard_name)

    async def create_leaderboard_in_db(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        date_value: datetime,
        task: LeaderboardTask,
        gpu: Optional[str] = None,
    ) -> bool:
        if gpu is None:
            # Ask the user to select GPUs
            view = GPUSelectionView(
                [gpu.name for gpu in GitHubGPU] + [gpu.name for gpu in ModalGPU]
            )

            await send_discord_message(
                interaction,
                "Please select GPUs for this leaderboard.",
                view=view,
                ephemeral=True,
            )

            await view.wait()
            selected_gpus = view.selected_gpus
        else:
            selected_gpus = [gpu]

        with self.bot.leaderboard_db as db:
            err = db.create_leaderboard(
                {
                    "name": leaderboard_name,
                    "deadline": date_value,
                    "task": task,
                    "gpu_types": selected_gpus,
                    "creator_id": interaction.user.id,
                }
            )

            if err:
                if "duplicate key" in err:
                    await send_discord_message(
                        interaction,
                        "Error: Tried to create a leaderboard "
                        f'"{leaderboard_name}" that already exists.',
                        ephemeral=True,
                    )
                else:
                    # Handle any other errors
                    logger.error(f"Error in leaderboard creation: {err}")
                    await send_discord_message(
                        interaction,
                        "Error in leaderboard creation.",
                        ephemeral=True,
                    )
                return False

            return True

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

    @discord.app_commands.describe(leaderboard_name="Name of the leaderboard")
    @discord.app_commands.autocomplete(leaderboard_name=leaderboard_name_autocomplete)
    async def delete_leaderboard(self, interaction: discord.Interaction, leaderboard_name: str):
        is_admin = await self.admin_check(interaction)
        is_creator = await self.creator_check(interaction)
        is_creator_of_leaderboard = await self.is_creator_check(interaction, leaderboard_name)

        if not (is_admin):
            if not is_creator:
                await send_discord_message(
                    interaction,
                    "You need the Leaderboard Creator role or the Leaderboard Admin role to use this command.",  # noqa: E501
                    ephemeral=True,
                )
                return
            if not is_creator_of_leaderboard:
                await send_discord_message(
                    interaction,
                    "You need to be the creator of the leaderboard to use this command.",
                    ephemeral=True,
                )
                return

        modal = DeleteConfirmationModal("leaderboard", leaderboard_name, self.bot.leaderboard_db)

        forum_channel = self.bot.get_channel(self.bot.leaderboard_forum_id)
        threads = [thread for thread in forum_channel.threads if thread.name == leaderboard_name]

        if threads:
            thread = threads[0]
            new_name = (
                f"{leaderboard_name} - archived at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            await thread.edit(name=new_name, archived=True)

        await interaction.response.send_modal(modal)
