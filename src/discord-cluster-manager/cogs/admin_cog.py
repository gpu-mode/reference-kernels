import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional, TypedDict

import discord
import env
import yaml
from consts import GitHubGPU, ModalGPU
from discord import app_commands
from discord.ext import commands
from leaderboard_db import leaderboard_name_autocomplete
from task import LeaderboardTask, make_task
from ui.misc import DeleteConfirmationModal, GPUSelectionView
from utils import (
    send_discord_message,
    setup_logging,
)

if TYPE_CHECKING:
    from ..bot import ClusterBot

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
    root = Path("examples")
    return [
        discord.app_commands.Choice(name=x.name, value=x.name) for x in root.iterdir() if x.is_dir()
    ]


class AdminCog(commands.Cog):
    def __init__(self, bot: "ClusterBot"):
        self.bot = bot

        # create-local should only be used for the development bot
        if self.bot.debug_mode:
            self.leaderboard_create_local = bot.admin_group.command(
                name="create-local",
                description="Create or replace a leaderboard from a local directory",
            )(self.leaderboard_create_local)

        self.delete_leaderboard = bot.admin_group.command(
            name="delete", description="Delete a leaderboard"
        )(self.delete_leaderboard)

        self.accept_jobs = bot.admin_group.command(
            name="start", description="Make the bot accept new submissions"
        )(self.start)

        self.reject_jobs = bot.admin_group.command(
            name="stop", description="Make the bot stop accepting new submissions"
        )(self.stop)

        self.update_problems = bot.admin_group.command(
            name="update-problems", description="Reload all problem definitions"
        )(self.update_problems)

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

    async def leaderboard_create_impl(  # noqa: C901
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        deadline: str,
        task: LeaderboardTask,
        gpus: Optional[str | list[str]],
    ):
        thread = None
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

        try:
            success = await self.create_leaderboard_in_db(
                interaction, leaderboard_name, date_value, task, gpus
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
    @discord.app_commands.autocomplete(leaderboard_name=leaderboard_name_autocomplete)
    async def delete_leaderboard(self, interaction: discord.Interaction, leaderboard_name: str):
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

    async def stop(self, interaction: discord.Interaction):
        is_admin = await self.admin_check(interaction)
        if not is_admin:
            await send_discord_message(
                interaction,
                "You need to have Admin permissions to run this command",
                ephemeral=True,
            )
            return

        self.bot.accepts_jobs = False
        await send_discord_message(
            interaction, "Bot will refuse all future submissions!", ephemeral=True
        )

    async def start(self, interaction: discord.Interaction):
        is_admin = await self.admin_check(interaction)
        if not is_admin:
            await send_discord_message(
                interaction,
                "You need to have Admin permissions to run this command",
                ephemeral=True,
            )
            return

        self.bot.accepts_jobs = True
        await send_discord_message(
            interaction, "Bot will accept submissions again!", ephemeral=True
        )

    @app_commands.describe(
        url="Repository from which to import problems",
        problem_set="Which problem set to load.",
        branch="Which branch to pull from",
    )
    async def update_problems(
        self,
        interaction: discord.Interaction,
        url: Optional[str] = None,
        problem_set: Optional[str] = None,
        branch: Optional[str] = None,
    ):
        is_admin = await self.admin_check(interaction)
        if not is_admin:
            await send_discord_message(
                interaction,
                "You need to have Admin permissions to run this command",
                ephemeral=True,
            )
            return

        if url is None:
            url = f"https://github.com/{env.PROBLEMS_REPO}.git"

        await interaction.response.defer(ephemeral=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            args = ["git", "clone", "--depth", "1"]
            if branch is not None:
                args += ["--single-branch", "--branch", branch]
            args += [url, temp_dir]
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

            # OK, we have the problems. Go over them one-by-one
            problem_dir = Path(temp_dir) / "problems"
            if problem_set is None:
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
                await self.update_competition(interaction, problem_set)

    async def _create_update_plan(  # noqa: C901
        self, interaction: discord.Interaction, competition: CompetitionData, root: Path
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
                new_task = make_task(source)

                # from the database, we get datetime with timezone,
                # so we need to convert here to enable comparison
                new_dl = self._parse_deadline(problem["deadline"])
                new_dl = new_dl.astimezone()
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

    async def update_competition(self, interaction: discord.Interaction, spec_file: Path):
        try:
            root = spec_file.parent
            with open(spec_file) as f:
                competition: CompetitionData = yaml.safe_load(f)

            header = f"Handling `{competition['name']}`..."
            await send_discord_message(interaction, header)

            update_list, create_list = await self._create_update_plan(
                interaction, competition, root
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
                    make_task(root / entry["directory"]),
                    entry["gpus"],
                )
                steps += "done\n"

            for entry in update_list:
                with self.bot.leaderboard_db as db:
                    db.update_leaderboard(
                        entry["name"], entry["deadline"], make_task(Path(entry["directory"]))
                    )

            header += " DONE"
            await interaction.edit_original_response(content=f"{header}\n\n{plan}\n\n{steps}")
        except Exception as e:
            logger.exception("Error updating problem set", exc_info=e)
