import asyncio
import random
from datetime import datetime
from io import StringIO
from typing import Optional

import discord
from consts import GitHubGPU, ModalGPU
from discord import app_commands
from discord.ext import commands
from leaderboard_db import leaderboard_name_autocomplete
from leaderboard_eval import cu_eval, py_eval
from ui.misc import DeleteConfirmationModal, GPUSelectionView
from ui.table import create_table
from utils import (
    extract_score,
    get_user_from_id,
    send_discord_message,
    setup_logging,
)

logger = setup_logging()


async def async_submit_github_job(
    interaction: discord.Interaction,
    leaderboard_name: str,
    script: discord.Attachment,
    github_command,
    reference_code,
    bot,
    submission_content,
    github_cog: commands.Cog,
    gpu: str,
):
    try:
        github_thread = await github_command.callback(
            github_cog,
            interaction,
            script,
            app_commands.Choice(
                name=gpu,
                value=GitHubGPU[gpu].value,
            ),
            reference_code=reference_code,
        )
    except discord.errors.NotFound as e:
        print(f"Webhook not found: {e}")
        await send_discord_message(interaction, "❌ The webhook was not found.")

    message_contents = [msg.content async for msg in github_thread.history(limit=None)]

    # Compute eval or submission score, call runner here.
    # TODO: Make this more robust later
    score = extract_score("".join(message_contents))

    with bot.leaderboard_db as db:
        db.create_submission(
            {
                "submission_name": script.filename,
                "submission_time": datetime.now(),
                "leaderboard_name": leaderboard_name,
                "code": submission_content,
                "user_id": interaction.user.id,
                "submission_score": score,
                "gpu_type": gpu,
            }
        )

    user_id = (
        interaction.user.global_name if interaction.user.nick is None else interaction.user.nick
    )

    await send_discord_message(
        interaction,
        f"Successfully ran on {gpu} using GitHub runners!\n"
        + f"Leaderboard '{leaderboard_name}'.\n"
        + f"Submission title: {script.filename}.\n"
        + f"Submission user: {user_id}.\n"
        + f"Runtime: {score:.9f} seconds.",
        ephemeral=True,
    )


class LeaderboardSubmitCog(app_commands.Group):
    def __init__(
        self,
        bot: commands.Bot,
    ):
        self.bot: commands.Bot = bot

        super().__init__(name="submit", description="Submit leaderboard data")

    # Parent command that defines global options
    @app_commands.describe(
        leaderboard_name="Name of the competition / kernel to optimize",
        script="The Python / CUDA script file to run",
    )
    @app_commands.autocomplete(leaderboard_name=leaderboard_name_autocomplete)
    # TODO: Modularize this so all the write functionality is in here. Haven't figured
    # a good way to do this yet.
    async def submit(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        script: discord.Attachment,
    ):
        pass

    @app_commands.command(name="modal", description="Submit leaderboard data for modal")
    @app_commands.describe(
        gpu_type="Choose the GPU type for Modal",
    )
    @app_commands.autocomplete(leaderboard_name=leaderboard_name_autocomplete)
    @app_commands.choices(
        gpu_type=[app_commands.Choice(name=gpu.value, value=gpu.value) for gpu in ModalGPU]
    )
    async def submit_modal(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        script: discord.Attachment,
        gpu_type: app_commands.Choice[str],
    ):
        try:
            # Read the template file
            submission_content = await script.read()

            # Call Modal runner
            modal_cog = self.bot.get_cog("ModalCog")

            if not all([modal_cog]):
                await send_discord_message(interaction, "❌ Required cogs not found!")
                return

            # Compute eval or submission score, call runner here.
            score = random.random()

            with self.bot.leaderboard_db as db:
                db.create_submission(
                    {
                        "submission_name": script.filename,
                        "submission_time": datetime.now(),
                        "leaderboard_name": leaderboard_name,
                        "code": submission_content,
                        "user_id": interaction.user.id,
                        "submission_score": score,
                        "gpu_type": gpu_type.name,
                    }
                )

            await send_discord_message(
                interaction,
                f"Ran on Modal. Leaderboard '{leaderboard_name}'.\n"
                + f"Submission title: {script.filename}.\n"
                + f"Submission user: {interaction.user.id}.\n"
                + f"Runtime: {score:.9f} seconds.",
                ephemeral=True,
            )
        except ValueError:
            await send_discord_message(
                interaction,
                "Invalid date format. Please use YYYY-MM-DD or YYYY-MM-DD HH:MM",
                ephemeral=True,
            )

    @app_commands.command(name="github", description="Submit leaderboard data for GitHub")
    @app_commands.autocomplete(leaderboard_name=leaderboard_name_autocomplete)
    async def submit_github(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        script: discord.Attachment,
    ):
        # Don't allow submissions if deadline is past
        with self.bot.leaderboard_db as db:
            leaderboard_item = db.get_leaderboard(leaderboard_name)

        # Read the template file
        submission_content = await script.read()

        try:
            submission_content = submission_content.decode()
        except UnicodeError:
            await send_discord_message(
                interaction, "Could not decode your file. Is it UTF-8?", ephemeral=True
            )
            return

        try:
            # Read and convert reference code
            reference_code = None
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
                    return

                if not leaderboard_item:
                    await send_discord_message(
                        interaction,
                        f"Leaderboard {leaderboard_name} not found.",
                        ephemeral=True,
                    )
                    return
                reference_code = leaderboard_item["reference_code"]
                gpus = db.get_leaderboard_gpu_types(leaderboard_name)

            if not interaction.response.is_done():
                await interaction.response.defer()

            # Call GH runner
            github_cog = self.bot.get_cog("GitHubCog")

            if not all([github_cog]):
                await send_discord_message(interaction, "❌ Required cogs not found!")
                return

            view = GPUSelectionView(gpus)

            await send_discord_message(
                interaction,
                f"Please select GPUs to submit for leaderboard: {leaderboard_name}.",
                view=view,
                ephemeral=True,
            )

            await view.wait()

            github_command = github_cog.run_github

            tasks = [
                async_submit_github_job(
                    interaction,
                    leaderboard_name,
                    script,
                    github_command,
                    reference_code,
                    self.bot,
                    reference_code,
                    github_cog,
                    gpu,
                )
                for gpu in view.selected_gpus
            ]
            await asyncio.gather(*tasks)

        except ValueError:
            await send_discord_message(
                interaction,
                "Invalid date format. Please use YYYY-MM-DD or YYYY-MM-DD HH:MM",
                ephemeral=True,
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

        self.delete_leaderboard = bot.leaderboard_group.command(
            name="delete", description="Delete a leaderboard"
        )(self.delete_leaderboard)

        self.get_leaderboard_submissions = bot.leaderboard_group.command(
            name="show", description="Get all submissions for a leaderboard"
        )(self.get_leaderboard_submissions)

        self.get_user_leaderboard_submissions = bot.leaderboard_group.command(
            name="show-personal", description="Get all your submissions for a leaderboard"
        )(self.get_user_leaderboard_submissions)

        self.get_leaderboard_references = bot.leaderboard_group.command(
            name="reference-code", description="Get leaderboard reference codes"
        )(self.get_leaderboard_references)

        self.get_leaderboard_eval = bot.leaderboard_group.command(
            name="eval-code", description="Get leaderboard evaluation codes"
        )(self.get_leaderboard_eval)

    # --------------------------------------------------------------------------
    # |                           HELPER FUNCTIONS                              |
    # --------------------------------------------------------------------------

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
            logger.error(str(e))
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

    # --------------------------------------------------------------------------
    # |                           COMMANDS                                      |
    # --------------------------------------------------------------------------

    async def get_leaderboards(self, interaction: discord.Interaction):
        """Display all leaderboards in a table format"""
        await interaction.response.defer(ephemeral=True)

        with self.bot.leaderboard_db as db:
            leaderboards = db.get_leaderboards()

        if not leaderboards:
            await send_discord_message(interaction, "No leaderboards found.", ephemeral=True)
            return

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

        await send_discord_message(
            interaction,
            "",
            embed=embed,
            view=view,
            ephemeral=True,
        )

    @app_commands.describe(leaderboard_name="Name of the leaderboard")
    @app_commands.autocomplete(leaderboard_name=leaderboard_name_autocomplete)
    async def get_leaderboard_references(
        self, interaction: discord.Interaction, leaderboard_name: str
    ):
        await interaction.response.defer(ephemeral=True)

        with self.bot.leaderboard_db as db:
            leaderboard_item = db.get_leaderboard(leaderboard_name)
            if not leaderboard_item:
                await send_discord_message(interaction, "Leaderboard not found.", ephemeral=True)
                return

        code = leaderboard_item["reference_code"]
        code_file = StringIO(code)
        language = "cpp" if "#include" in code else "py"

        ref_code = discord.File(
            fp=code_file, filename=f"{leaderboard_name}_reference_code.{language}"
        )

        message = (
            f"**Reference Code for {leaderboard_name} in {language}**\n"
            f"*If you want to display the evaluation code, run `/leaderboard eval-code {language}`*"
        )

        await send_discord_message(interaction, message, ephemeral=True, file=ref_code)

    @app_commands.describe(language="Language of the evaluation code [cpp, python]")
    @app_commands.choices(
        language=[app_commands.Choice(name=lang, value=lang) for lang in ["cpp", "python"]]
    )
    async def get_leaderboard_eval(self, interaction: discord.Interaction, language: str):
        await interaction.response.defer(ephemeral=True)

        if language == "cpp":
            eval_code = cu_eval
        else:
            eval_code = py_eval

        code_file = StringIO(eval_code)
        ref_code = discord.File(fp=code_file, filename=f"leaderboard_eval.{language}")

        await send_discord_message(
            interaction,
            f"**Evaluation Code for language: {language}**\n",
            file=ref_code,
            ephemeral=True,
        )

    @discord.app_commands.describe(
        leaderboard_name="Name of the leaderboard",
        deadline="Competition deadline in the form: 'Y-m-d'",
        reference_code="Reference implementation of kernel. Also includes eval code.",
    )
    async def leaderboard_create(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        deadline: str,
        reference_code: discord.Attachment,
    ):
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

        # Ask the user to select GPUs
        view = GPUSelectionView([gpu.name for gpu in GitHubGPU])

        await send_discord_message(
            interaction,
            "Please select GPUs for this leaderboard.",
            view=view,
            ephemeral=True,
        )

        await view.wait()

        try:
            # Read the template file
            template_content = await reference_code.read()

            with self.bot.leaderboard_db as db:
                err = db.create_leaderboard(
                    {
                        "name": leaderboard_name,
                        "deadline": date_value,
                        "reference_code": template_content.decode("utf-8"),
                        "gpu_types": view.selected_gpus,
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
                    return

            await send_discord_message(
                interaction,
                f"Leaderboard '{leaderboard_name}'.\n"
                + f"Reference code: {reference_code}. Submission deadline: {date_value}",
                ephemeral=True,
            )

        except Exception as e:
            logger.error(f"Error in leaderboard creation: {e}")
            # Handle any other errors
            await send_discord_message(
                interaction,
                "Error in leaderboard creation.",
                ephemeral=True,
            )

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
        modal = DeleteConfirmationModal("leaderboard", leaderboard_name, self.bot.leaderboard_db)
        await interaction.response.send_modal(modal)
