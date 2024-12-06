import discord
from discord import app_commands
from discord.ext import commands
from datetime import datetime

from typing import TYPE_CHECKING
from consts import GitHubGPU, ModalGPU

if TYPE_CHECKING:
    from bot import ClusterBot


class LeaderboardSubmitCog(app_commands.Group):
    def __init__(self):
        super().__init__(name="submit", description="Submit leaderboard data")

    # Parent command that defines global options
    @app_commands.describe(
        leaderboard_name="Name of the competition / kernel to optimize",
        script="The Python / CUDA script file to run",
        dtype="dtype (e.g. FP32, BF16, FP4) that the input and output expects.",
        shape="Data input shape as a tuple",
    )
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
    async def modal(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        script: discord.Attachment,
        gpu_type: app_commands.Choice[str],
        dtype: app_commands.Choice[str] = "fp32",
        shape: app_commands.Choice[str] = None,
    ):
        await interaction.response.send_message(
            f"Submitted modal data: GPU Type={gpu_type}"
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
            app_commands.Choice(name=gpu.value, value=gpu.value) for gpu in GitHubGPU
        ]
    )
    async def github(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        script: discord.Attachment,
        gpu_type: app_commands.Choice[str],
        dtype: app_commands.Choice[str] = "fp32",
        shape: app_commands.Choice[str] = None,
    ):
        await interaction.response.send_message(
            f"Submitted GitHub data: GPU Type={gpu_type}"
        )


class LeaderboardCog(commands.Cog):
    def __init__(self, bot):
        self.bot: ClusterBot = bot
        self.get_leaderboards = bot.leaderboard_group.command(name="get")(
            self.get_leaderboards
        )
        self.leaderboard_create = bot.leaderboard_group.command(
            name="create", description="Create a new leaderboard"
        )(self.leaderboard_create)

        # self.leaderboard_submit = bot.leaderboard_group.command(
        #     name="submit", description="Submit a file to the leaderboard"
        # )(self.leaderboard_submit)

        bot.leaderboard_group.add_command(LeaderboardSubmitCog())

        self.get_leaderboard_submissions = bot.leaderboard_group.command(
            name="submissions", description="Get all submissions for a leaderboard"
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
                db.create_leaderboard({
                    "name": leaderboard_name,
                    "deadline": date_value,
                    "reference_code": template_content.decode("utf-8"),
                })

            await interaction.response.send_message(
                f"Leaderboard '{leaderboard_name}'. Submission deadline: {date_value}",
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
            leaderboard_id = db.get_leaderboard_id(leaderboard_name)
            if not leaderboard_id:
                await interaction.response.send_message(
                    "Leaderboard not found.", ephemeral=True
                )
                return

            submissions = db.get_leaderboard_submissions(leaderboard_id)  # Add dtype

        if not submissions:
            await interaction.response.send_message(
                "No submissions found.", ephemeral=True
            )
            return

        # Create embed
        embed = discord.Embed(
            title="Leaderboard Submissions", color=discord.Color.blue()
        )

        for submission in submissions:
            embed.add_field(
                name=f"{submission['user_id']}: submission['submission_name']",
                value=f"Submission time: {submission['submission_time']}",
                inline=False,
            )

        await interaction.response.send_message(embed=embed)
