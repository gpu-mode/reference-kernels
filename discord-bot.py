from dotenv import load_dotenv
import os
import time
from datetime import datetime, timezone
import requests
import discord
from discord import app_commands
import asyncio
import logging
import zipfile
import subprocess
import argparse
from enum import Enum
import modal
from github import Github

# Configure logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def init_environment():
    load_dotenv()
    logger.info("Environment variables loaded")

init_environment()

class GPUType(Enum):
    NVIDIA = "nvidia_workflow.yml"
    AMD = "amd_workflow.yml"

class SchedulerType(Enum):
    GITHUB = "github"
    MODAL = "modal"
    SLURM = "slurm"

def get_github_branch_name():
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{u}'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().split('/', 1)[1]
    except subprocess.CalledProcessError:
        return 'main'

# Validate environment
required_env_vars = ['DISCORD_TOKEN', 'GITHUB_TOKEN', 'GITHUB_REPO']
for var in required_env_vars:
    if not os.getenv(var):
        raise ValueError(f"{var} not found")

class ClusterBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

        # Simple ping command
        @self.tree.command(name="ping")
        async def ping(interaction: discord.Interaction):
            await interaction.response.send_message("pong")

        # Admin command to resync
        @self.tree.command(name="resync")
        async def resync(interaction: discord.Interaction):
            if interaction.user.guild_permissions.administrator:
                try:
                    await interaction.response.defer(ephemeral=True)
                    # Clear and resync
                    self.tree.clear_commands(guild=interaction.guild)
                    await self.tree.sync(guild=interaction.guild)
                    commands = await self.tree.fetch_commands(guild=interaction.guild)
                    await interaction.followup.send(
                        f"Resynced commands:\n" + 
                        "\n".join([f"- /{cmd.name}" for cmd in commands]),
                        ephemeral=True
                    )
                except Exception as e:
                    await interaction.followup.send(f"Error: {str(e)}", ephemeral=True)
            else:
                await interaction.response.send_message(
                    "You need administrator permissions to use this command",
                    ephemeral=True
                )

        # Create the run command group
        self.run_group = app_commands.Group(name="run", description="Run jobs on different platforms")

        # Modal subcommand
        @self.run_group.command(name="modal")
        @app_commands.describe(
            script="The Python script file to run",
            gpu_type="Choose the GPU type for Modal"
        )
        @app_commands.choices(
            gpu_type=[
                app_commands.Choice(name="NVIDIA T4", value="t4"),
                # app_commands.Choice(name="NVIDIA A10G", value="a10g")
            ]
        )
        async def run_modal(
            interaction: discord.Interaction,
            script: discord.Attachment,
            gpu_type: app_commands.Choice[str]
        ):
            if not script.filename.endswith('.py'):
                await interaction.response.send_message("Please provide a Python (.py) file", ephemeral=True)
                return

            thread = await interaction.channel.create_thread(
                name=f"Modal Job ({gpu_type.name}) - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                auto_archive_duration=1440
            )

            await interaction.response.send_message(f"Created thread {thread.mention} for your Modal job", ephemeral=True)
            await thread.send(f"Processing `{script.filename}` with {gpu_type.name}...")

            try:
                script_content = (await script.read()).decode('utf-8')
                await thread.send("Running on Modal...")
                result = await trigger_modal_run(script_content, script.filename)
                await thread.send(f"```\nModal execution result:\n{result}\n```")
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}", exc_info=True)
                await thread.send(f"Error processing request: {str(e)}")

        # GitHub subcommand
        @self.run_group.command(name="github")
        @app_commands.describe(
            script="The Python script file to run",
            gpu_type="Choose the GPU type for GitHub Actions"
        )
        @app_commands.choices(
            gpu_type=[
                app_commands.Choice(name="NVIDIA", value="nvidia"),
                app_commands.Choice(name="AMD", value="amd")
            ]
        )
        async def run_github(
            interaction: discord.Interaction,
            script: discord.Attachment,
            gpu_type: app_commands.Choice[str]
        ):
            if not script.filename.endswith('.py'):
                await interaction.response.send_message("Please provide a Python (.py) file", ephemeral=True)
                return

            thread = await interaction.channel.create_thread(
                name=f"GitHub Job ({gpu_type.name}) - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                auto_archive_duration=1440
            )

            await interaction.response.send_message(f"Created thread {thread.mention} for your GitHub job", ephemeral=True)
            await thread.send(f"Processing `{script.filename}` with {gpu_type.name}...")

            try:
                script_content = (await script.read()).decode('utf-8')
                selected_gpu = GPUType.AMD if gpu_type.value == "amd" else GPUType.NVIDIA
                run_id = await trigger_github_action(script_content, script.filename, selected_gpu)
                
                if run_id:
                    await thread.send(f"GitHub Action triggered! Run ID: {run_id}\nMonitoring progress...")
                    status, logs, url = await check_workflow_status(run_id, thread)
                    
                    await thread.send(f"Training completed with status: {status}")
                    
                    if len(logs) > 1900:
                        chunks = [logs[i:i+1900] for i in range(0, len(logs), 1900)]
                        for i, chunk in enumerate(chunks):
                            await thread.send(f"```\nLogs (part {i+1}/{len(chunks)}):\n{chunk}\n```")
                    else:
                        await thread.send(f"```\nLogs:\n{logs}\n```")
                    
                    if url:
                        await thread.send(f"View the full run at: {url}")
                else:
                    await thread.send("Failed to trigger GitHub Action. Please check the configuration.")
            
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}", exc_info=True)
                await thread.send(f"Error processing request: {str(e)}")

        # Add the run group to the command tree
        self.tree.add_command(self.run_group)

    async def setup_hook(self):
        try:
            guild_id = os.getenv('DISCORD_CLUSTER_STAGING_ID')
            if guild_id:
                guild = discord.Object(id=int(guild_id))
                
                # Clear existing commands
                self.tree.clear_commands(guild=guild)
                logger.info("Cleared existing commands")
                
                # Copy global commands to guild and sync
                self.tree.copy_global_to(guild=guild)
                await self.tree.sync(guild=guild)
                
                # Verify commands
                commands = await self.tree.fetch_commands(guild=guild)
                logger.info(f"Synced commands: {[cmd.name for cmd in commands]}")
        except Exception as e:
            logger.error(f"Failed to sync commands: {e}")

client = ClusterBot()

@client.event
async def on_ready():
    logger.info(f'Logged in as {client.user}')
    for guild in client.guilds:
        try:
            if globals().get('args') and args.debug:
                await guild.me.edit(nick="Cluster Bot (Staging)")
            else:
                await guild.me.edit(nick="Cluster Bot")
        except Exception as e:
            logger.warning(f'Failed to update nickname in guild {guild.name}: {e}')

async def trigger_modal_run(script_content: str, filename: str) -> str:
    logger.info("Attempting to trigger Modal run")
    try:
        from modal_runner import run_script, modal_app
        with modal.enable_output():
            with modal_app.run():
                result = run_script.remote(script_content)
            return result
    except Exception as e:
        logger.error(f"Error in trigger_modal_run: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

async def trigger_github_action(script_content, filename, gpu_type):
    logger.info(f"Attempting to trigger GitHub action for {gpu_type.name} GPU")
    gh = Github(os.getenv('GITHUB_TOKEN'))
    repo = gh.get_repo(os.getenv('GITHUB_REPO'))
    
    try:
        trigger_time = datetime.now(timezone.utc)
        workflow_file = gpu_type.value
        workflow = repo.get_workflow(workflow_file)
        
        success = workflow.create_dispatch(get_github_branch_name(), {
            'script_content': script_content,
            'filename': filename
        })
        
        if success:
            await asyncio.sleep(2)
            runs = list(workflow.get_runs())
            
            for run in runs:
                if run.created_at.replace(tzinfo=timezone.utc) > trigger_time:
                    return run.id
        return None
            
    except Exception as e:
        logger.error(f"Error in trigger_github_action: {str(e)}", exc_info=True)
        return None

async def download_artifact(run_id):
    logger.info(f"Attempting to download artifacts for run {run_id}")
    gh = Github(os.getenv('GITHUB_TOKEN'))
    repo = gh.get_repo(os.getenv('GITHUB_REPO'))
    
    try:
        run = repo.get_workflow_run(run_id)
        artifacts = run.get_artifacts()
        
        for artifact in artifacts:
            if artifact.name == 'training-artifacts':
                url = artifact.archive_download_url
                headers = {'Authorization': f'token {os.getenv("GITHUB_TOKEN")}'}
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    with open('training.log.zip', 'wb') as f:
                        f.write(response.content)
                    
                    with zipfile.ZipFile('training.log.zip') as z:
                        log_file = next((f for f in z.namelist() if f.endswith('training.log')), None)
                        if log_file:
                            with z.open(log_file) as f:
                                logs = f.read().decode('utf-8')
                        else:
                            logs = "training.log file not found in artifact"
                    
                    os.remove('training.log.zip')
                    return logs
                else:
                    return f"Failed to download artifact. Status code: {response.status_code}"
        
        return "No training artifacts found"
    except Exception as e:
        return f"Error downloading artifacts: {str(e)}"

async def check_workflow_status(run_id, thread):
    logger.info(f"Starting to monitor workflow status for run {run_id}")
    gh = Github(os.getenv('GITHUB_TOKEN'))
    repo = gh.get_repo(os.getenv('GITHUB_REPO'))
    
    while True:
        try:
            run = repo.get_workflow_run(run_id)
            
            if run.status == "completed":
                logs = await download_artifact(run_id)
                return run.conclusion, logs, run.html_url
            
            await thread.send(f"Workflow still running... Status: {run.status}\nLive view: {run.html_url}")
            await asyncio.sleep(60)
        except Exception as e:
            return "error", str(e), None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Discord Cluster Bot')
    parser.add_argument('--debug', action='store_true', help='Run in debug/staging mode')
    args = parser.parse_args()

    logger.info("Starting bot...")
    if args.debug:
        token = os.getenv('DISCORD_DEBUG_TOKEN')
        if not token:
            raise ValueError("DISCORD_DEBUG_TOKEN not found")
    else:
        token = os.getenv('DISCORD_TOKEN')
    
    client.run(token)