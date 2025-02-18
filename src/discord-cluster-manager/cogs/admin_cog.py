import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import discord
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

        self.leaderboard_create = bot.admin_group.command(
            name="create", description="Create a new leaderboard"
        )(self.leaderboard_create)

        # create-local should only be used for the development bot
        if self.bot.debug_mode:
            self.leaderboard_create_local = bot.admin_group.command(
                name="create-local",
                description="Create or replace a leaderboard from a local directory",
            )(self.leaderboard_create_local)

        self.delete_leaderboard = bot.admin_group.command(
            name="delete", description="Delete a leaderboard"
        )(self.delete_leaderboard)

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
