import random
from datetime import datetime

import discord
from consts import GitHubGPU, ModalGPU
from discord import app_commands
from discord.ext import commands
from utils import extract_score, get_user_from_id


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
        dtype="dtype (e.g. FP32, BF16, FP4) that the input and output expects.",
        shape="Data input shape as a tuple",
    )
    # TODO: Modularize this so all the write functionality is in here. Haven't figured
    # a good way to do this yet.
    async def submit(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        script: discord.Attachment,
        dtype: app_commands.Choice[str] = None,
        shape: app_commands.Choice[str] = None,
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
        dtype: app_commands.Choice[str] = "fp32",
        shape: app_commands.Choice[str] = None,
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
        dtype: app_commands.Choice[str] = "fp32",
        shape: app_commands.Choice[str] = None,
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
            print(github_command)
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

            user_id = interaction.user.global_name if interaction.user.nick is None else interaction.user.nick
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
        try:
            # Try parsing with time first
            try:
                date_value = datetime.strptime(deadline, "%Y-%m-%d %H:%M")
            except ValueError:
                # If that fails, try parsing just the date (will set time to 00:00)
                date_value = datetime.strptime(deadline, "%Y-%m-%d")

            # Read the template file
            template_content = await reference_code.read()

            with self.bot.leaderboard_db as db:
                print(
                    leaderboard_name,
                    type(date_value),
                    type(template_content.decode("utf-8")),
                )
                db.create_leaderboard({
                    "name": leaderboard_name,
                    "deadline": date_value,
                    "reference_code": template_content.decode("utf-8"),
                })

            await interaction.response.send_message(
                f"Leaderboard '{leaderboard_name}' created.\n"
                + f"Reference code: {reference_code}.\n"
                + f"Submission deadline: {date_value}",
                ephemeral=True,
            )
        except ValueError:
            await interaction.response.send_message(
                "Invalid date format. Please use YYYY-MM-DD or YYYY-MM-DD HH:MM",
                ephemeral=True,
            )

    @discord.app_commands.describe(leaderboard_name="Name of the leaderboard")
    async def get_leaderboard_submissions(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        dtype: app_commands.Choice[str] = "fp32",
    ):
        with self.bot.leaderboard_db as db:
            # TODO: query that gets leaderboard id given leaderboard name
            leaderboard_id = db.get_leaderboard(leaderboard_name)["id"]
            if not leaderboard_id:
                await interaction.response.send_message(
                    "Leaderboard not found.", ephemeral=True
                )
                return

            # submissions = db.get_leaderboard_submissions(leaderboard_id)  # Add dtype
            submissions = db.get_leaderboard_submissions(leaderboard_name)  # Add dtype

        if not submissions:
            await interaction.response.send_message(
                "No submissions found.", ephemeral=True
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
            print("members", interaction.guild.members)
            print(user_id)

            embed.add_field(
                name=f"{user_id}: {submission['submission_name']}",
                value=f"Submission speed: {submission['submission_score']}",
                inline=False,
            )

        await interaction.response.send_message(embed=embed)
