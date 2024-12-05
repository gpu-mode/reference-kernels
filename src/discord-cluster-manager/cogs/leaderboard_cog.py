import discord
from datetime import datetime

from discord.ext import commands

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot import ClusterBot


class LeaderboardSubmitCog(discord.app_commands.Group):
    def __init__(self):
        super().__init__(name="submit", description="Submit leaderboard data")

    # Parent command that defines global options
    @discord.app_commands.command(name="submit", description="Submit leaderboard data")
    @discord.app_commands.describe(
        global_arg="A global argument that will propagate to subcommands"
    )
    async def submit(
        self,
        interaction: discord.Interaction,
        global_arg: str,  # Global argument for the parent command
    ):
        pass

    ## MODAL SUBCOMMAND
    @discord.app_commands.command(
        name="modal", description="Submit leaderboard data for modal"
    )
    @discord.app_commands.describe(
        modal_x="Value for field X",
        modal_y="Value for field Y",
        modal_z="Value for field Z",
    )
    async def modal(
        self,
        interaction: discord.Interaction,
        global_arg: str,
        modal_x: str,
        modal_y: str,
        modal_z: str,
    ):
        await interaction.response.send_message(
            f"Submitted modal data: X={modal_x}, Y={modal_y}, Z={modal_z}"
        )

    ### GITHUB SUBCOMMAND
    @discord.app_commands.command(
        name="github", description="Submit leaderboard data for GitHub"
    )
    @discord.app_commands.describe(
        github_x="Value for field X",
        github_yint="Value for field Y",
        github_z="Value for field Z",
    )
    async def github(
        self,
        interaction: discord.Interaction,
        global_arg: str,
        github_x: str,
        github_yint: int,
        github_z: str,
    ):
        await interaction.response.send_message(
            f"Submitted GitHub data: X={github_x}, Y_int={github_yint}, Z={github_z}]"
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
        name="Name of the leaderboard",
        date="Date in YYYY-MM-DD format (time HH:MM is optional)",
        template_file="Template file to upload",
    )
    async def leaderboard_create(
        self,
        interaction: discord.Interaction,
        name: str,
        date: str,
        template_file: discord.Attachment,
    ):
        try:
            # Try parsing with time first
            try:
                date_value = datetime.strptime(date, "%Y-%m-%d %H:%M")
            except ValueError:
                # If that fails, try parsing just the date (will set time to 00:00)
                date_value = datetime.strptime(date, "%Y-%m-%d")

            # Read the template file
            template_content = await template_file.read()

            with self.bot.leaderboard_db as db:
                db.create_leaderboard({
                    "name": name,
                    "deadline": date_value,
                    "template_code": template_content.decode("utf-8"),
                })

            await interaction.response.send_message(
                f"Leaderboard '{name}'. Submission deadline: {date_value}",
                ephemeral=True,
            )
        except ValueError:
            await interaction.response.send_message(
                "Invalid date format. Please use YYYY-MM-DD or YYYY-MM-DD HH:MM",
                ephemeral=True,
            )

    @discord.app_commands.describe(leaderboard_name="Name of the leaderboard")
    async def get_leaderboard_submissions(
        self, interaction: discord.Interaction, leaderboard_name: str
    ):
        with self.bot.leaderboard_db as db:
            leaderboard_id = db.get_leaderboard_id(leaderboard_name)
            if not leaderboard_id:
                await interaction.response.send_message(
                    "Leaderboard not found.", ephemeral=True
                )
                return

            submissions = db.get_leaderboard_submissions(leaderboard_id)

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
