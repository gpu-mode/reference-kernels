import json
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Optional, TypedDict

import discord
import yaml
from discord import app_commands
from discord.ext import commands, tasks

from kernelbot.discord_utils import (
    leaderboard_name_autocomplete,
    send_discord_message,
    with_error_handling,
)
from kernelbot.env import env
from kernelbot.ui.misc import ConfirmationView, DeleteConfirmationModal, GPUSelectionView
from libkernelbot.consts import GitHubGPU, ModalGPU
from libkernelbot.leaderboard_db import LeaderboardDoesNotExist, LeaderboardItem, SubmissionItem
from libkernelbot.task import LeaderboardDefinition, make_task_definition
from libkernelbot.utils import (
    KernelBotError,
    setup_logging,
)

if TYPE_CHECKING:
    from kernelbot.main import ClusterBot

logger = setup_logging()


class ProblemData(TypedDict):
    name: str
    directory: str
    deadline: str
    gpus: list[str]


class CompetitionData(TypedDict):
    name: str
    description: str
    deadline: str
    problems: list[ProblemData]


async def leaderboard_dir_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> list[discord.app_commands.Choice[str]]:
    """Return leaderboard names that match the current typed name"""
    root = Path(env.PROBLEM_DEV_DIR)
    return [
        discord.app_commands.Choice(name=x.name, value=x.name) for x in root.iterdir() if x.is_dir()
    ]


# ensure valid serialization
def serialize(obj: object):
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


class AdminCog(commands.Cog):
    def __init__(self, bot: "ClusterBot"):
        self.bot = bot

        # create-local should only be used for the development kernelbot
        if self.bot.debug_mode:
            self.leaderboard_create_local = bot.admin_group.command(
                name="create-local",
                description="Create or replace a leaderboard from a local directory",
            )(self.leaderboard_create_local)

        self.delete_leaderboard = bot.admin_group.command(
            name="delete-leaderboard", description="Delete a leaderboard"
        )(self.delete_leaderboard)

        self.delete_submission = bot.admin_group.command(
            name="delete-submission", description="Delete a submission"
        )(self.delete_submission)

        self.accept_jobs = bot.admin_group.command(
            name="start", description="Make the kernelbot accept new submissions"
        )(self.start)

        self.reject_jobs = bot.admin_group.command(
            name="stop", description="Make the kernelbot stop accepting new submissions"
        )(self.stop)

        self.update_problems = bot.admin_group.command(
            name="update-problems", description="Reload all problem definitions"
        )(self.update_problems)

        self.show_bot_stats = bot.admin_group.command(
            name="show-stats", description="Show stats for the kernelbot"
        )(self.show_bot_stats)

        self.resync = bot.admin_group.command(
            name="resync", description="Trigger re-synchronization of slash commands"
        )(self.resync)

        self.get_submission_by_id = bot.admin_group.command(
            name="get-submission", description="Retrieve one of past submissions"
        )(self.get_submission_by_id)

        self.get_user_names = bot.admin_group.command(
            name="get-user-names", description="Get user names"
        )(self.get_user_names)

        self.update_user_names = bot.admin_group.command(
            name="update-user-names", description="Update user names"
        )(self.update_user_names)

        self.set_forum_ids = bot.admin_group.command(
            name="set-forum-ids", description="Sets forum IDs"
        )(self.set_forum_ids)

        self._scheduled_cleanup_temp_users.start()

    # --------------------------------------------------------------------------
    # |                           HELPER FUNCTIONS                              |
    # --------------------------------------------------------------------------

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

    @discord.app_commands.describe(
        directory="Directory of the kernel definition. Also used as the leaderboard's name",
        gpu="The GPU to submit to. Leave empty for interactive selection/multiple GPUs",
    )
    @app_commands.autocomplete(directory=leaderboard_dir_autocomplete)
    @app_commands.choices(
        gpu=[app_commands.Choice(name=gpu.name, value=gpu.value) for gpu in GitHubGPU]
        + [app_commands.Choice(name=gpu.name, value=gpu.value) for gpu in ModalGPU]
    )
    @with_error_handling
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

        directory = Path(env.PROBLEM_DEV_DIR) / directory
        assert directory.resolve().is_relative_to(Path.cwd() / env.PROBLEM_DEV_DIR)
        definition = make_task_definition(directory)

        # clearly mark this leaderboard as development-only
        leaderboard_name = directory.name + "-dev"

        # create-local overwrites existing leaderboard
        with self.bot.leaderboard_db as db:
            try:
                old_lb = db.get_leaderboard(leaderboard_name)
            except LeaderboardDoesNotExist:
                old_lb = None
            db.delete_leaderboard(leaderboard_name, force=True)

        # get existing forum thread or create new one
        forum_channel = self.bot.get_channel(self.bot.leaderboard_forum_id)
        forum_thread = None
        if old_lb:
            forum_id = old_lb["forum_id"]
            forum_thread = await self.bot.fetch_channel(forum_id)

        if forum_thread is None:
            forum_thread = await forum_channel.create_thread(
                name=leaderboard_name,
                content=f"# Test Leaderboard: {leaderboard_name}\n\n",
                auto_archive_duration=10080,  # 7 days
            )
            forum_id = forum_thread.thread.id
        else:
            await forum_thread.send("Leaderboard was updated")

        if await self.create_leaderboard_in_db(
            interaction,
            leaderboard_name,
            datetime.now(timezone.utc) + timedelta(days=365),
            definition=definition,
            forum_id=forum_id,
            gpu=gpu.value if gpu else None,
        ):
            await send_discord_message(
                interaction,
                f"Leaderboard '{leaderboard_name}' created.",
            )

    def _parse_deadline(self, deadline: str):
        # Try parsing with time first
        try:
            return datetime.strptime(deadline, "%Y-%m-%d %H:%M")
        except ValueError:
            try:
                return datetime.strptime(deadline, "%Y-%m-%d")
            except ValueError as ve:
                logger.error(f"Value Error: {str(ve)}", exc_info=True)
        return None

    def _leaderboard_opening_message(
        self, leaderboard_name: str, deadline: datetime, description: str
    ):
        return f"""
        # New Leaderboard: {leaderboard_name}\n
        **Deadline**: {deadline.strftime('%Y-%m-%d %H:%M')}\n
        {description}\n
        Submit your entries using `/leaderboard submit ranked` in the submissions channel.\n
        Good luck to all participants! 🚀 <@&{self.bot.leaderboard_participant_role_id}>"""

    async def leaderboard_create_impl(  # noqa: C901
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        deadline: str,
        definition: LeaderboardDefinition,
        gpus: Optional[str | list[str]],
    ):
        if len(leaderboard_name) > 95:
            await send_discord_message(
                interaction,
                "Leaderboard name is too long. Please keep it under 95 characters.",
                ephemeral=True,
            )
            return

        date_value = self._parse_deadline(deadline)
        if date_value is None:
            await send_discord_message(
                interaction,
                "Invalid date format. Please use YYYY-MM-DD or YYYY-MM-DD HH:MM",
                ephemeral=True,
            )

        if date_value < datetime.now():
            await send_discord_message(
                interaction,
                f"Deadline {date_value} has already passed.",
                ephemeral=True,
            )
            return

        forum_channel = self.bot.get_channel(self.bot.leaderboard_forum_id)
        forum_thread = None
        try:
            forum_thread = await forum_channel.create_thread(
                name=leaderboard_name,
                content=self._leaderboard_opening_message(
                    leaderboard_name, date_value, definition.description
                ),
                auto_archive_duration=10080,  # 7 days
            )

            success = await self.create_leaderboard_in_db(
                interaction, leaderboard_name, date_value, definition, forum_thread.thread.id, gpus
            )
            if not success:
                await forum_thread.delete()
                return

            await send_discord_message(
                interaction,
                f"Leaderboard '{leaderboard_name}'.\n"
                + f"Submission deadline: {date_value}"
                + f"\nForum thread: {forum_thread.thread.mention}",
            )
            return

        except discord.Forbidden:
            await send_discord_message(
                interaction,
                "Error: Bot doesn't have permission to create forum threads."
                " Leaderboard was not created.",
                ephemeral=True,
            )
        except discord.HTTPException:
            await send_discord_message(
                interaction,
                "Error creating forum thread. Leaderboard was not created.",
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
        if forum_thread is not None:
            await forum_thread.delete()

        with self.bot.leaderboard_db as db:  # Cleanup in case lb was created
            db.delete_leaderboard(leaderboard_name)

    async def create_leaderboard_in_db(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        date_value: datetime,
        definition: LeaderboardDefinition,
        forum_id: int,
        gpu: Optional[str | list[str]] = None,
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
        elif isinstance(gpu, str):
            selected_gpus = [gpu]
        else:
            selected_gpus = gpu

        with self.bot.leaderboard_db as db:
            try:
                db.create_leaderboard(
                    name=leaderboard_name,
                    deadline=date_value,
                    definition=definition,
                    gpu_types=selected_gpus,
                    creator_id=interaction.user.id,
                    forum_id=forum_id,
                )
            except KernelBotError as e:
                await send_discord_message(
                    interaction,
                    str(e),
                    ephemeral=True,
                )
                return False
            return True

    @discord.app_commands.describe(leaderboard_name="Name of the leaderboard")
    @discord.app_commands.autocomplete(leaderboard_name=leaderboard_name_autocomplete)
    @with_error_handling
    async def delete_leaderboard(
        self, interaction: discord.Interaction, leaderboard_name: str, force: bool = False
    ):
        is_admin = await self.admin_check(interaction)
        is_creator = await self.creator_check(interaction)
        is_creator_of_leaderboard = await self.is_creator_check(interaction, leaderboard_name)

        if not is_admin:
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

        modal = DeleteConfirmationModal(
            "leaderboard", leaderboard_name, self.bot.leaderboard_db, force=force
        )

        forum_channel = self.bot.get_channel(self.bot.leaderboard_forum_id)

        with self.bot.leaderboard_db as db:
            forum_id = db.get_leaderboard(leaderboard_name)["forum_id"]
        threads = [thread for thread in forum_channel.threads if thread.id == forum_id]

        if threads:
            thread = threads[0]
            new_name = (
                f"{leaderboard_name} - archived at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            await thread.edit(name=new_name, archived=True)

        await interaction.response.send_modal(modal)

    @discord.app_commands.describe(submission="ID of the submission to delete")
    @with_error_handling
    async def delete_submission(self, interaction: discord.Interaction, submission: int):
        is_admin = await self.admin_check(interaction)

        if not is_admin:
            await send_discord_message(
                interaction,
                "You need to be Admin to use this command.",
                ephemeral=True,
            )
            return

        with self.bot.leaderboard_db as db:
            sub = db.get_submission_by_id(submission_id=submission)

        if sub is None:
            await send_discord_message(
                interaction,
                f"No submission of id `{submission}`.",
                ephemeral=True,
            )
            return

        msg, files = self._make_submission_message(submission, sub)

        async def do_delete():
            with self.bot.leaderboard_db as db:
                db.delete_submission(submission_id=submission)

            await send_discord_message(
                interaction,
                f"💥 Submission `{submission}` has been **deleted**.",
                ephemeral=True,
            )

        async def no_delete():
            await send_discord_message(
                interaction,
                f"💾 Submission `{submission}` has **not** been deleted.",
                ephemeral=True,
            )

        confirm = ConfirmationView(
            confirm_text="Delete",
            confirm_callback=do_delete,
            reject_text="Keep",
            reject_callback=no_delete,
        )
        await send_discord_message(
            interaction, "# Attention\nYou are about to **delete** the following submission:\n"
        )
        await send_discord_message(interaction, msg, files=files)
        await send_discord_message(
            interaction,
            "💂 Please confirm!",
            view=confirm,
            ephemeral=True,
        )

    @with_error_handling
    async def stop(self, interaction: discord.Interaction):
        is_admin = await self.admin_check(interaction)
        if not is_admin:
            await send_discord_message(
                interaction,
                "You need to have Admin permissions to run this command",
                ephemeral=True,
            )
            return

        self.bot.backend.accepts_jobs = False
        await send_discord_message(
            interaction, "Bot will refuse all future submissions!", ephemeral=True
        )

    @with_error_handling
    async def start(self, interaction: discord.Interaction):
        is_admin = await self.admin_check(interaction)
        if not is_admin:
            await send_discord_message(
                interaction,
                "You need to have Admin permissions to run this command",
                ephemeral=True,
            )
            return

        self.bot.backend.accepts_jobs = True
        await send_discord_message(
            interaction, "Bot will accept submissions again!", ephemeral=True
        )

    @app_commands.describe(
        problem_set="Which problem set to load.",
        repository_name="Name of the repository to load problems from (in format: user/repo)",
        branch="Which branch to pull from",
    )
    @with_error_handling
    async def update_problems(
        self,
        interaction: discord.Interaction,
        repository_name: Optional[str] = None,
        problem_set: Optional[str] = None,
        branch: Optional[str] = "main",
        force: bool = False,
    ):
        is_admin = await self.admin_check(interaction)
        if not is_admin:
            await send_discord_message(
                interaction,
                "You need to have Admin permissions to run this command",
                ephemeral=True,
            )
            return

        if "/" in branch:
            raise KernelBotError(f"branch names with slashes (`{branch}`) are not supported.")

        repository_name = repository_name or env.PROBLEMS_REPO
        url = f"https://github.com/{repository_name}/archive/{branch}.zip"
        folder_name = repository_name.split("/")[-1] + "-" + branch

        await interaction.response.defer(ephemeral=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            args = ["wget", "-O", temp_dir + "/problems.zip", url]
            try:
                subprocess.check_call(args, encoding="utf-8")
            except subprocess.CalledProcessError as E:
                logger.exception("could not git clone problems repo: %s", E.stderr, exc_info=E)
                # admin-only command, we can send error messages as ephemeral
                msg = f"could not git clone `{url}`:\nstdout: {E.stdout}\nstderr: {E.stderr}"
                await send_discord_message(
                    interaction,
                    msg,
                    ephemeral=True,
                )
                return

            args = ["unzip", temp_dir + "/problems.zip", "-d", temp_dir]
            try:
                subprocess.check_call(args, encoding="utf-8")
            except subprocess.CalledProcessError as E:
                logger.exception("could not unzip problems repo: %s", E.stderr, exc_info=E)
                # admin-only command, we can send error messages as ephemeral
                msg = f"could not unzip `{temp_dir}/problems.zip`:\nstdout: {E.stdout}\nstderr: {E.stderr}"  # noqa: E501
                await send_discord_message(
                    interaction,
                    msg,
                    ephemeral=True,
                )
                return

            # OK, we have the problems. Go over them one-by-one
            problem_dir = Path(temp_dir) / folder_name / "problems"
            if problem_set is None:
                if force:
                    await send_discord_message(
                        interaction,
                        "Cannot use force without specifying problem set",
                        ephemeral=True,
                    )
                    return
                for competition in problem_dir.glob("*.yaml"):
                    await self.update_competition(interaction, competition)
            else:
                problem_set = problem_dir / f"{problem_set}.yaml"
                if not problem_set.exists():
                    msg = f"Could not find problem set {problem_set} in repository {url}.\n"
                    msg += "Available options:\n\n* "
                    msg += "\n* ".join([f.stem for f in problem_dir.glob("*.yaml")])
                    await send_discord_message(
                        interaction,
                        msg,
                        ephemeral=True,
                    )
                    return
                await self.update_competition(interaction, problem_set, force)

    async def _create_update_plan(  # noqa: C901
        self,
        interaction: discord.Interaction,
        competition: CompetitionData,
        root: Path,
        force: bool,
    ):
        update_list = []
        create_list = []

        with self.bot.leaderboard_db as db:
            leaderboards = db.get_leaderboards()
        leaderboards = {lb["name"]: lb for lb in leaderboards}

        # TODO lots of QoL improvements here: scope problem names, problem versioning
        for problem in competition["problems"]:
            source = root / problem["directory"]
            name = problem["name"]
            if not source.exists():
                await send_discord_message(
                    interaction,
                    f"Directory `{source}` for problem `{name}` does not exist, skipping.",
                )
                continue

            # check if that leaderboard already exists
            if name in leaderboards:
                # check for differences
                old = leaderboards[name]  # type: LeaderboardItem
                new_def = make_task_definition(source)
                new_task = new_def.task

                # from the database, we get datetime with timezone,
                # so we need to convert here to enable comparison
                new_dl = self._parse_deadline(problem["deadline"])
                new_dl = new_dl.astimezone(timezone.utc)
                if old["deadline"] != new_dl:
                    pass
                elif old["gpu_types"] != problem["gpus"]:
                    await send_discord_message(
                        interaction,
                        "Changing GPU types of an existing problem is currently not possible",
                    )
                    continue
                elif old["task"] != new_task:
                    ot = old["task"]
                    # TODO improve this! force should require confirmation.
                    if force:
                        update_list.append(problem)
                        continue
                    # now look what precisely has changed. For the moment, disallow anything
                    # that would require us to do more careful task versioning;
                    # we can only change things that have no bearing on existing
                    # runs (like description and templates)
                    if ot.files != new_task.files:
                        file_list = set.symmetric_difference(
                            set(ot.files.keys()), set(new_task.files)
                        )
                        if len(file_list) != 0:
                            await send_discord_message(
                                interaction,
                                f"Adding or removing task files of existing problem `{name}`"
                                f" is currently not possible. File list difference: {file_list}",
                            )
                        else:
                            diff_files = {
                                key for key in ot.files if ot.files[key] != new_task.files[key]
                            }
                            await send_discord_message(
                                interaction,
                                f"Changing task files of existing problem `{name}`"
                                f" is currently not possible. Changed files: {diff_files}",
                            )
                        continue
                    if ot.config != new_task.config:
                        await send_discord_message(
                            interaction,
                            "Changing task config of an existing problem is currently not possible",
                        )
                        continue

                    if ot.lang != new_task.lang:
                        await send_discord_message(
                            interaction,
                            "Changing language of an existing problem is currently not possible",
                        )
                        continue

                    if ot.benchmarks != new_task.benchmarks:
                        await send_discord_message(
                            interaction,
                            "Changing benchmarks of an existing problem is currently not possible",
                        )
                        continue

                else:
                    # no changes
                    continue
                update_list.append(problem)
            else:
                create_list.append(problem)

        return update_list, create_list

    async def update_competition(
        self, interaction: discord.Interaction, spec_file: Path, force: bool = False
    ):
        try:
            root = spec_file.parent
            with open(spec_file) as f:
                competition: CompetitionData = yaml.safe_load(f)

            header = f"Handling `{competition['name']}`..."
            await send_discord_message(interaction, header)

            update_list, create_list = await self._create_update_plan(
                interaction, competition, root, force
            )

            # OK, now we know what we want to do
            plan = ""
            if len(update_list) > 0:
                lst = "\n * ".join(x["name"] for x in update_list)
                plan += f"The following leaderboards will be updated:\n * {lst}\n"
            if len(create_list):
                lst = "\n * ".join(x["name"] for x in create_list)
                plan += f"The following new leaderboards will be created:\n * {lst}\n"

            if plan == "":
                plan = "Everything is up-to-date\n"

            await interaction.edit_original_response(content=f"{header}\n\n{plan}")

            steps = ""
            # TODO require confirmation here!
            for entry in create_list:
                steps += f"Creating {entry['name']}... "
                await interaction.edit_original_response(content=f"{header}\n\n{plan}\n\n{steps}")
                await self.leaderboard_create_impl(
                    interaction,
                    entry["name"],
                    entry["deadline"],
                    make_task_definition(root / entry["directory"]),
                    entry["gpus"],
                )
                steps += "done\n"

            for entry in update_list:
                with self.bot.leaderboard_db as db:
                    task = make_task_definition(root / entry["directory"])
                    db.update_leaderboard(
                        entry["name"], self._parse_deadline(entry["deadline"]), task
                    )
                    new_lb: LeaderboardItem = db.get_leaderboard(entry["name"])

                forum_id = new_lb["forum_id"]
                try:
                    forum_thread = await self.bot.fetch_channel(forum_id)
                    if forum_thread and forum_thread.starter_message:
                        await forum_thread.starter_message.edit(
                            content=self._leaderboard_opening_message(
                                entry["name"], new_lb["deadline"], task.description
                            )
                        )
                except (discord.errors.NotFound, discord.errors.HTTPException):
                    logger.warning(
                        "Could not find forum thread %s for lb %s", forum_id, entry["name"]
                    )
                    pass

            header += " DONE"
            await interaction.edit_original_response(content=f"{header}\n\n{plan}\n\n{steps}")
        except Exception as e:
            logger.exception("Error updating problem set", exc_info=e)

    @with_error_handling
    @discord.app_commands.describe(last_day_only="Only show stats for the last day")
    async def show_bot_stats(self, interaction: discord.Interaction, last_day_only: bool):
        is_admin = await self.admin_check(interaction)
        if not is_admin:
            await send_discord_message(
                interaction,
                "You need to have Admin permissions to run this command",
                ephemeral=True,
            )
            return

        with self.bot.leaderboard_db as db:
            stats = db.generate_stats(last_day_only)
            msg = """```"""
            for k, v in stats.items():
                msg += f"\n{k} = {v}"
            msg += "\n```"
            await send_discord_message(interaction, msg, ephemeral=True)

    @with_error_handling
    async def resync(self, interaction: discord.Interaction):
        """Admin command to resync slash commands"""
        logger.info("Resyncing commands")
        if interaction.user.guild_permissions.administrator:
            try:
                await interaction.response.defer()
                # Clear and resync
                self.bot.tree.clear_commands(guild=interaction.guild)
                await self.bot.tree.sync(guild=interaction.guild)
                commands = await self.bot.tree.fetch_commands(guild=interaction.guild)
                await send_discord_message(
                    interaction,
                    "Resynced commands:\n" + "\n".join([f"- /{cmd.name}" for cmd in commands]),
                )
            except Exception as e:
                logger.error(f"Error in resync command: {str(e)}", exc_info=True)
                await send_discord_message(interaction, f"Error: {str(e)}")
        else:
            await send_discord_message(
                interaction, "You need administrator permissions to use this command"
            )

    # admin version of this command; less restricted
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
        if sub is None:
            await send_discord_message(
                interaction, f"Submission {submission_id} does not exist", ephemeral=True
            )
            return

        msg, files = self._make_submission_message(submission_id, sub)
        await send_discord_message(interaction, msg, ephemeral=True, files=files)

    def _make_submission_message(self, submission_id: int, sub: SubmissionItem):
        msg = f"# Submission {submission_id}\n"
        msg += f"submitted by {sub['user_id']} on {sub['submission_time']}"
        msg += f" to leaderboard `{sub['leaderboard_name']}`."
        if not sub["done"]:
            msg += "\n*Submission is still running!*\n"

        file = discord.File(fp=StringIO(sub["code"]), filename=sub["file_name"])

        if len(sub["runs"]) > 0:
            msg += "\nRuns:\n"
        for run in sub["runs"]:
            msg += f" * {run['mode']} on {run['runner']}: "
            if run["score"] is not None and run["passed"]:
                msg += f"{run['score']}"
            else:
                msg += "pass" if run["passed"] else "fail"
            msg += "\n"

        run_results = discord.File(
            fp=StringIO(json.dumps(sub["runs"], default=serialize, indent=2)), filename="runs.json"
        )

        return msg, [file, run_results]

    @tasks.loop(minutes=10)
    async def _scheduled_cleanup_temp_users(self):
        with self.bot.leaderboard_db as db:
            db.cleanup_temp_users()
        logger.info("Temporary users cleanup completed")

    ####################################################################################################################
    #            MIGRATION COMMANDS --- TO BE DELETED LATER
    ####################################################################################################################

    async def get_user_names(self, interaction: discord.Interaction):
        """Get a mapping of user IDs to their names"""
        if not await self.admin_check(interaction):
            await send_discord_message(
                interaction,
                "You need to have Admin permissions to run this command",
                ephemeral=True,
            )
            return
        await interaction.response.defer()
        try:
            with self.bot.leaderboard_db as db:
                db.cursor.execute("""
                    SELECT DISTINCT user_id
                    FROM leaderboard.submission
                """)
                user_ids = [row[0] for row in db.cursor.fetchall()]

            user_mapping = {}
            for user_id in user_ids:
                try:
                    discord_id = int(user_id)
                    user = await self.bot.fetch_user(discord_id)
                    user_mapping[user_id] = user.global_name or user.name
                except (ValueError, discord.NotFound, discord.HTTPException) as e:
                    logger.error(f"Error fetching user {user_id}: {str(e)}")
                    user_mapping[user_id] = "Unknown User"

            with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as temp_file:
                json.dump(user_mapping, temp_file, indent=2)

            await interaction.followup.send(
                content="Here's the mapping of user IDs to names:",
                file=discord.File(temp_file.name, filename="user_mapping.json"),
            )

            import os

            os.unlink(temp_file.name)

        except Exception as e:
            error_message = f"Error generating user mapping: {str(e)}"
            logger.error(error_message, exc_info=True)
            await send_discord_message(interaction, error_message)

    @app_commands.describe(attachment="The JSON file containing user ID to name mapping")
    async def update_user_names(
        self, interaction: discord.Interaction, attachment: discord.Attachment
    ):
        """Update the database with user names from a JSON file"""
        if not await self.admin_check(interaction):
            await send_discord_message(
                interaction,
                "You need to have Admin permissions to run this command",
                ephemeral=True,
            )
            return
        await interaction.response.defer()

        try:
            if not attachment.filename.endswith(".json"):
                await send_discord_message(
                    interaction, "Please attach a JSON file with .json extension."
                )
                return

            json_content = await attachment.read()
            user_mapping = json.loads(json_content)

            updated_count = 0
            with self.bot.leaderboard_db as db:
                for user_id, user_name in user_mapping.items():
                    try:
                        # First check if user exists in user_info
                        db.cursor.execute(
                            """
                            SELECT 1 FROM leaderboard.user_info WHERE id = %s LIMIT 1
                            """,
                            (user_id,),
                        )
                        if db.cursor.fetchone():
                            # Update existing user
                            db.cursor.execute(
                                """
                                UPDATE leaderboard.user_info
                                SET user_name = %s
                                WHERE id = %s
                                """,
                                (user_name, user_id),
                            )
                        else:
                            # Insert new user
                            db.cursor.execute(
                                """
                                INSERT INTO leaderboard.user_info (id, user_name)
                                VALUES (%s, %s)
                                """,
                                (user_id, user_name),
                            )
                        updated_count += db.cursor.rowcount
                    except Exception as e:
                        logger.error(f"Error updating user {user_id}: {str(e)}")

                db.connection.commit()

            await send_discord_message(
                interaction,
                f"Successfully updated {updated_count} user records with names.",
            )

        except json.JSONDecodeError:
            await send_discord_message(
                interaction, "Invalid JSON format in the attached file.", ephemeral=True
            )
        except Exception as e:
            error_message = f"Error updating database with user names: {str(e)}"
            logger.error(error_message, exc_info=True)
            await send_discord_message(interaction, error_message, ephemeral=True)

    async def set_forum_ids(self, interaction: discord.Interaction):
        try:
            with self.bot.leaderboard_db as db:
                db.cursor.execute(
                    """
                    SELECT id, name
                    FROM leaderboard.leaderboard
                    WHERE forum_id = -1
                    """,
                )

                for id, name in db.cursor.fetchall():
                    # search forum threads
                    forum_channel = self.bot.get_channel(self.bot.leaderboard_forum_id)
                    threads = [thread for thread in forum_channel.threads if thread.name == name]
                    if len(threads) == 0:
                        # is it an archived thread?
                        threads = [
                            thread
                            async for thread in forum_channel.archived_threads()
                            if thread.name == name
                        ]
                    if len(threads) != 1:
                        await send_discord_message(
                            interaction, f"Could not set forum thread for {name}", ephemeral=True
                        )
                        continue
                    thread = threads[0]
                    db.cursor.execute(
                        """
                        UPDATE leaderboard.leaderboard
                        SET forum_id = %s
                        WHERE id = %s
                        """,
                        (thread.id, id),
                    )

                db.connection.commit()
                await send_discord_message(
                    interaction,
                    "Successfully updated forum ids.",
                )
        except Exception as e:
            error_message = f"Error updating forum ids: {str(e)}"
            logger.error(error_message, exc_info=True)
            await send_discord_message(interaction, error_message, ephemeral=True)
