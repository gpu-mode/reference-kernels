from dotenv import load_dotenv
from github import Github
import os
import time
from datetime import datetime, timezone
import requests
import discord
import asyncio
import logging
import zipfile
import subprocess
import argparse
from enum import Enum

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

class GPUType(Enum):
    NVIDIA = "nvidia_workflow.yml"
    AMD = "amd_workflow.yml"

def get_gpu_type(message_content):
    """
    Determine GPU type based on message content
    """
    if "AMD" in message_content.upper():
        return GPUType.AMD
    return GPUType.NVIDIA  # Default to NVIDIA if not specified

def get_github_branch_name():
    """
    Runs a git command to determine the remote branch name, to be used in the GitHub Workflow
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{u}'],
            capture_output=True,
            text=True,
            check=True
        )
        logging.info(f"Remote branch found: {result.stdout.strip().split('/', 1)[1]}")
        return result.stdout.strip().split('/', 1)[1]
    except subprocess.CalledProcessError:
        logging.warning("Could not determine remote branch, falling back to 'main'")
        return 'main'

# Validate environment variables
required_env_vars = ['DISCORD_TOKEN', 'GITHUB_TOKEN', 'GITHUB_REPO']
for var in required_env_vars:
    if not os.getenv(var):
        logger.error(f"{var} not found in environment variables")
        raise ValueError(f"{var} not found")

logger.info(f"Using GitHub repo: {os.getenv('GITHUB_REPO')}")

# Bot setup with minimal intents
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

async def trigger_github_action(script_content, filename, gpu_type):
    """
    Triggers the GitHub action with custom script contents and filename
    """
    logger.info(f"Attempting to trigger GitHub action for {gpu_type.name} GPU")
    gh = Github(os.getenv('GITHUB_TOKEN'))
    repo = gh.get_repo(os.getenv('GITHUB_REPO'))
    
    try:
        trigger_time = datetime.now(timezone.utc)
        workflow_file = gpu_type.value
        logger.info(f"Looking for workflow '{workflow_file}' in repo {os.getenv('GITHUB_REPO')}")
        
        workflow = repo.get_workflow(workflow_file)
        logger.info(f"Found workflow, attempting to dispatch for {gpu_type.name}")
        
        success = workflow.create_dispatch(get_github_branch_name(), {
            'script_content': script_content,
            'filename': filename
        })
        logger.info(f"Workflow dispatch result: {success}")
        
        if success:
            await asyncio.sleep(2)
            runs = list(workflow.get_runs())
            logger.info(f"Found {len(runs)} total runs")
            
            for run in runs:
                logger.info(f"Checking run {run.id} created at {run.created_at}")
                if run.created_at.replace(tzinfo=timezone.utc) > trigger_time:
                    logger.info(f"Found matching run with ID: {run.id}")
                    return run.id
            
            logger.warning("No matching runs found after trigger")
            return None
            
    except Exception as e:
        logger.error(f"Error in trigger_github_action: {str(e)}", exc_info=True)
        return None

async def download_artifact(run_id):
    """
    Downloads the training log artifact from the workflow run
    """
    logger.info(f"Attempting to download artifacts for run {run_id}")
    gh = Github(os.getenv('GITHUB_TOKEN'))
    repo = gh.get_repo(os.getenv('GITHUB_REPO'))
    
    try:
        run = repo.get_workflow_run(run_id)
        artifacts = run.get_artifacts()
        logger.info(f"Found {artifacts.totalCount} artifacts")
        
        for artifact in artifacts:
            logger.info(f"Found artifact: {artifact.name}")
            if artifact.name == 'training-artifacts':
                url = artifact.archive_download_url
                headers = {'Authorization': f'token {os.getenv("GITHUB_TOKEN")}'}
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    logger.info("Successfully downloaded artifact")
                    with open('training.log.zip', 'wb') as f:
                        f.write(response.content)
                    
                    with zipfile.ZipFile('training.log.zip') as z:
                        # Updated to handle potential different file paths
                        log_file = None
                        for file in z.namelist():
                            if file.endswith('training.log'):
                                log_file = file
                                break
                        
                        if log_file:
                            with z.open(log_file) as f:
                                logs = f.read().decode('utf-8')
                        else:
                            logs = "training.log file not found in artifact"
                    
                    os.remove('training.log.zip')
                    return logs
                else:
                    logger.error(f"Failed to download artifact. Status code: {response.status_code}")
                    return f"Failed to download artifact. Status code: {response.status_code}"
        
        logger.warning("No training-artifacts found")
        return "No training artifacts found. Available artifacts: " + ", ".join([a.name for a in artifacts])
    except Exception as e:
        logger.error(f"Error in download_artifact: {str(e)}", exc_info=True)
        return f"Error downloading artifacts: {str(e)}"

async def check_workflow_status(run_id, thread):
    """
    Monitors the GitHub Action workflow status and updates Discord thread
    """
    logger.info(f"Starting to monitor workflow status for run {run_id}")
    gh = Github(os.getenv('GITHUB_TOKEN'))
    repo = gh.get_repo(os.getenv('GITHUB_REPO'))
    
    while True:
        try:
            run = repo.get_workflow_run(run_id)
            logger.info(f"Current status: {run.status}")
            
            if run.status == "completed":
                logger.info("Workflow completed, downloading artifacts")
                logs = await download_artifact(run_id)
                return run.conclusion, logs, run.html_url
            
            await thread.send(f"Workflow still running... Status: {run.status}\nLive view: {run.html_url}")
            await asyncio.sleep(30)
        except Exception as e:
            logger.error(f"Error in check_workflow_status: {str(e)}", exc_info=True)
            return "error", str(e), None

@client.event
async def on_ready():
    logger.info(f'Logged in as {client.user}')
    for guild in client.guilds:
        try:
            if globals().get('args') and args.debug:
                await guild.me.edit(nick="Cluster Bot (Staging)")
            else:
                await guild.me.edit(nick="Cluster Bot")
            logger.info(f'Updated nickname in guild: {guild.name}')
        except Exception as e:
            logger.warning(f'Failed to update nickname in guild {guild.name}: {e}')

@client.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == client.user:
        return

    # Check if the bot is mentioned and there's an attachment
    if client.user in message.mentions:
        logger.info(f"Bot mentioned in message with {len(message.attachments)} attachments")
        if message.attachments:
            for attachment in message.attachments:
                logger.info(f"Processing attachment: {attachment.filename}")
                if attachment.filename.endswith('.py'):
                    # Determine GPU type from message
                    gpu_type = get_gpu_type(message.content)
                    logger.info(f"Selected {gpu_type.name} GPU for processing")
                    
                    # Create a thread directly from the original message
                    thread = await message.create_thread(
                        name=f"{gpu_type.name} Training Job - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        auto_archive_duration=1440  # Archive after 24 hours of inactivity
                    )
                    
                    # Send initial message in the thread
                    await thread.send(f"Found {attachment.filename}! Starting training process on {gpu_type.name} GPU...")
                    
                    try:
                        # Download the file content
                        logger.info(f"Downloading {attachment.filename} content")
                        script_content = await attachment.read()
                        script_content = script_content.decode('utf-8')
                        logger.info(f"Successfully read {attachment.filename} content")
                        
                        # Trigger GitHub Action
                        run_id = await trigger_github_action(script_content, attachment.filename, gpu_type)
                        
                        await asyncio.sleep(10)
                        
                        if run_id:
                            logger.info(f"Successfully triggered {gpu_type.name} workflow with run ID: {run_id}")
                            await thread.send(f"GitHub Action triggered successfully on {gpu_type.name}! Run ID: {run_id}\nMonitoring progress...")
                            
                            # Monitor the workflow
                            status, logs, url = await check_workflow_status(run_id, thread)
                            
                            # Send results back to Discord thread
                            await thread.send(f"Training completed with status: {status}")
                            
                            # Split logs if they're too long for Discord's message limit
                            if len(logs) > 1900:
                                chunks = [logs[i:i+1900] for i in range(0, len(logs), 1900)]
                                for i, chunk in enumerate(chunks):
                                    await thread.send(f"```\nLogs (part {i+1}/{len(chunks)}):\n{chunk}\n```")
                            else:
                                await thread.send(f"```\nLogs:\n{logs}\n```")
                            
                            if url:
                                await thread.send(f"View the full run at: {url}")
                        else:
                            logger.error(f"Missing run_id. Failed to trigger GitHub Action for {gpu_type.name}")
                            await thread.send(f"Failed to trigger GitHub Action for {gpu_type.name}. Please check the configuration.")
                    
                    except Exception as e:
                        logger.error(f"Error processing request: {str(e)}", exc_info=True)
                        await thread.send(f"Error processing request: {str(e)}")
                    
                    break

            if not any(att.filename.endswith('.py') for att in message.attachments):
                await message.reply("Please attach a Python file to your message. Include 'AMD' in your message to use AMD GPU, otherwise NVIDIA will be used.")

# Run the bot
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Discord Cluster Bot')
    parser.add_argument('--debug', action='store_true', help='Run in debug/staging mode')
    args = parser.parse_args()

    logger.info("Starting bot...")
    if args.debug:
        logger.info("Running in debug mode")
        token = os.getenv('DISCORD_DEBUG_TOKEN')
        if not token:
            logger.error("DISCORD_DEBUG_TOKEN not found in environment variables")
            raise ValueError("DISCORD_DEBUG_TOKEN not found")
    else:
        token = os.getenv('DISCORD_TOKEN')
    
    client.run(token)
