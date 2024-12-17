import random
from datetime import datetime

import discord
from consts import GitHubGPU, ModalGPU
from discord import Interaction, SelectOption, app_commands, ui
from discord.ext import commands
from utils import extract_score, get_user_from_id, setup_logging

logger = setup_logging()


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
    @app_commands.choices(
        gpu_type=[
            app_commands.Choice(name=gpu.value, value=gpu.value) for gpu in ModalGPU
        ]
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
                await interaction.response.send_message("❌ Required cogs not found!")
                return

            # Compute eval or submission score, call runner here.
            score = random.random()

            with self.bot.leaderboard_db as db:
                db.create_submission({
                    "submission_name": script.filename,
                    "submission_time": datetime.now(),
                    "leaderboard_name": leaderboard_name,
                    "code": submission_content,
                    "user_id": interaction.user.id,
                    "submission_score": score,
                })

            await interaction.response.send_message(
                f"Ran on Modal. Leaderboard '{leaderboard_name}'.\n"
                + f"Submission title: {script.filename}.\n"
                + f"Submission user: {interaction.user.id}.\n"
                + f"Runtime: {score} ms",
                ephemeral=True,
            )
        except ValueError:
            await interaction.response.send_message(
                "Invalid date format. Please use YYYY-MM-DD or YYYY-MM-DD HH:MM",
                ephemeral=True,
            )

    ### GITHUB SUBCOMMAND
    @app_commands.command(
        name="github", description="Submit leaderboard data for GitHub"
    )
    @app_commands.describe(
        gpu_type="Choose the GPU type for Github Runners",
    )
    @app_commands.choices(
        gpu_type=[
            app_commands.Choice(name=gpu.name, value=gpu.value) for gpu in GitHubGPU
        ]
    )
    async def submit_github(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        script: discord.Attachment,
        gpu_type: app_commands.Choice[str],
    ):
        # Read the template file
        submission_content = await script.read()

        try:
            submission_content = submission_content.decode()
        except UnicodeError:
            await interaction.response.send_message(
                "Could not decode your file. Is it UTF-8?", ephemeral=True
            )
            return

        try:
            # Read and convert reference code
            reference_code = None
            with self.bot.leaderboard_db as db:
                # TODO: query that gets reference code given leaderboard name
                leaderboard_item = db.get_leaderboard(leaderboard_name)
                if not leaderboard_item:
                    await interaction.response.send_message(
                        f"Leaderboard {leaderboard_name} not found.", ephemeral=True
                    )
                    return
                reference_code = leaderboard_item["reference_code"]

            if not interaction.response.is_done():
                await interaction.response.defer()

            # Call GH runner
            github_cog = self.bot.get_cog("GitHubCog")

            if not all([github_cog]):
                await interaction.followup.send("❌ Required cogs not found!")
                return

            github_command = github_cog.run_github
            try:
                github_thread = await github_command.callback(
                    github_cog,
                    interaction,
                    script,
                    gpu_type,
                    reference_code=reference_code,
                    use_followup=True,
                )
            except discord.errors.NotFound as e:
                print(f"Webhook not found: {e}")
                await interaction.followup.send("❌ The webhook was not found.")

            message_contents = [
                msg.content async for msg in github_thread.history(limit=None)
            ]

            # Compute eval or submission score, call runner here.
            # TODO: Make this more robust later
            score = extract_score("".join(message_contents))

            with self.bot.leaderboard_db as db:
                db.create_submission({
                    "submission_name": script.filename,
                    "submission_time": datetime.now(),
                    "leaderboard_name": leaderboard_name,
                    "code": submission_content,
                    "user_id": interaction.user.id,
                    "submission_score": score,
                })

            user_id = (
                interaction.user.global_name
                if interaction.user.nick is None
                else interaction.user.nick
            )
            await interaction.followup.send(
                "Successfully ran on GitHub runners!\n"
                + f"Leaderboard '{leaderboard_name}'.\n"
                + f"Submission title: {script.filename}.\n"
                + f"Submission user: {user_id}\n"
                + f"Runtime: {score} ms\n",
            )
        except ValueError:
            await interaction.followup.send(
                "Invalid date format. Please use YYYY-MM-DD or YYYY-MM-DD HH:MM",
                ephemeral=True,
            )


class GPUSelectionView(ui.View):
    def __init__(self, available_gpus: list[str]):
        super().__init__()

        # Add the Select Menu with the list of GPU options
        select = ui.Select(
            placeholder="Select GPUs for this leaderboard...",
            options=[SelectOption(label=gpu, value=gpu) for gpu in available_gpus],
            min_values=1,  # Minimum number of selections
            max_values=len(available_gpus),  # Maximum number of selections
        )
        select.callback = self.select_callback
        self.add_item(select)

    async def select_callback(self, interaction: Interaction):
        # Retrieve the selected options
        select = interaction.data["values"]
        self.selected_gpus = select
        await interaction.response.send_message(
            f"Selected GPUs: {', '.join(self.selected_gpus)}",
            ephemeral=True,
        )
        self.stop()


class LeaderboardCog(commands.Cog):
    def __init__(self, bot):
        self.bot: commands.Bot = bot
        self.get_leaderboards = bot.leaderboard_group.command(name="get")(
            self.get_leaderboards
        )
        self.leaderboard_create = bot.leaderboard_group.command(
            name="create", description="Create a new leaderboard"
        )(self.leaderboard_create)

        bot.leaderboard_group.add_command(LeaderboardSubmitCog(bot))

        self.get_leaderboard_submissions = bot.leaderboard_group.command(
            name="show", description="Get all submissions for a leaderboard"
        )(self.get_leaderboard_submissions)

    async def get_leaderboards(self, interaction: discord.Interaction):
        """Display all leaderboards in a table format"""
        await interaction.response.defer()

        with self.bot.leaderboard_db as db:
            leaderboards = db.get_leaderboards()

        if not leaderboards:
            await interaction.followup.send("No leaderboards found.", ephemeral=True)
            return

        # Create embed
        embed = discord.Embed(title="Active Leaderboards", color=discord.Color.blue())

        # Add fields for each leaderboard
        for lb in leaderboards:
            deadline_str = lb["deadline"].strftime("%Y-%m-%d %H:%M")
            embed.add_field(
                name=lb["name"], value=f"Deadline: {deadline_str}", inline=False
            )

        await interaction.followup.send(embed=embed)

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
                await interaction.response.send_message(
                    "Invalid date format. Please use YYYY-MM-DD or YYYY-MM-DD HH:MM",
                    ephemeral=True,
                )
                return

        # Ask the user to select GPUs
        view = GPUSelectionView([gpu.name for gpu in GitHubGPU])

        if interaction.response.is_done():
            await interaction.followup.send(
                "Please select GPUs for this leaderboard.",
                view=view,
                ephemeral=True,
            )
        else:
            await interaction.response.send_message(
                "Please select GPUs for this leaderboard.",
                view=view,
                ephemeral=True,
            )

        await view.wait()

        try:
            # Read the template file
            template_content = await reference_code.read()

            with self.bot.leaderboard_db as db:
                err = db.create_leaderboard({
                    "name": leaderboard_name,
                    "deadline": date_value,
                    "reference_code": template_content.decode("utf-8"),
                    "gpu_types": view.selected_gpus,
                })

                if err:
                    if "duplicate key" in err:
                        await interaction.followup.send(
                            f'Error: Tried to create a leaderboard "{leaderboard_name}" that already exists.',
                            ephemeral=True,
                        )
                    else:
                        # Handle any other errors
                        logger.error(f"Error in leaderboard creation: {err}")
                        await interaction.followup.send(
                            "Error in leaderboard creation.",
                            ephemeral=True,
                        )
                    return

            await interaction.followup.send(
                f"Leaderboard '{leaderboard_name}'.\n"
                + f"Reference code: {reference_code}. Submission deadline: {date_value}",
                ephemeral=True,
            )

        except Exception as e:
            logger.error(f"Error in leaderboard creation: {e}")
            # Handle any other errors
            await interaction.followup.send(
                "Error in leaderboard creation.",
                ephemeral=True,
            )
            

    @discord.app_commands.describe(leaderboard_name="Name of the leaderboard")
    async def get_leaderboard_submissions(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
    ):
        try:
            with self.bot.leaderboard_db as db:
                # TODO: query that gets leaderboard id given leaderboard name
                leaderboard_id = db.get_leaderboard(leaderboard_name)["id"]
                if not leaderboard_id:
                    await interaction.response.send_message(
                        f'Leaderboard "{leaderboard_name}" not found.', ephemeral=True
                    )
                    return

                submissions = db.get_leaderboard_submissions(leaderboard_name)

            if not submissions:
                await interaction.response.send_message(
                    f'No submissions found for "{leaderboard_name}".', ephemeral=True
                )
                return

            # Create embed
            embed = discord.Embed(
                title=f'Leaderboard Submissions for "{leaderboard_name}"',
                color=discord.Color.blue(),
            )

            for submission in submissions:
                user_id = await get_user_from_id(
                    submission["user_id"], interaction, self.bot
                )

                embed.add_field(
                    name=f"{user_id}: {submission['submission_name']}",
                    value=f"Submission speed: {submission['submission_score']}",
                    inline=False,
                )

            await interaction.response.send_message(embed=embed)
        except Exception as e:
            logger.error(str(e))
            if "'NoneType' object is not subscriptable" in str(e):
                await interaction.response.send_message(
                    f"The leaderboard '{leaderboard_name}' doesn't exist.",
                    ephemeral=True,
                )
            else:
                await interaction.response.send_message(
                    "An unknown error occurred.", ephemeral=True
                )
