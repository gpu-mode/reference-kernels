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
if not os.getenv('DISCORD_TOKEN'):
    logger.error("DISCORD_TOKEN not found in environment variables")
    raise ValueError("DISCORD_TOKEN not found")
if not os.getenv('GITHUB_TOKEN'):
    logger.error("GITHUB_TOKEN not found in environment variables")
    raise ValueError("GITHUB_TOKEN not found")
if not os.getenv('GITHUB_REPO'):
    logger.error("GITHUB_REPO not found in environment variables")
    raise ValueError("GITHUB_REPO not found")

logger.info(f"Using GitHub repo: {os.getenv('GITHUB_REPO')}")

# Bot setup with minimal intents
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


async def trigger_github_action(script_content):
    """
    Triggers the GitHub action with custom train.py contents
    """
    logger.info("Attempting to trigger GitHub action")
    gh = Github(os.getenv('GITHUB_TOKEN'))
    repo = gh.get_repo(os.getenv('GITHUB_REPO'))
    
    try:
        trigger_time = datetime.now(timezone.utc)
        logger.info(f"Looking for workflow 'train_workflow.yml' in repo {os.getenv('GITHUB_REPO')}")
        
        workflow = repo.get_workflow("train_workflow.yml")
        logger.info("Found workflow, attempting to dispatch")
        
        success = workflow.create_dispatch(get_github_branch_name(), {'script_content': script_content})
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
            if artifact.name == 'training-logs':
                url = artifact.archive_download_url
                headers = {'Authorization': f'token {os.getenv("GITHUB_TOKEN")}'}
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    logger.info("Successfully downloaded artifact")
                    with open('training.log.zip', 'wb') as f:
                        f.write(response.content)
                    
                    with zipfile.ZipFile('training.log.zip') as z:
                        with z.open('training.log') as f:
                            logs = f.read().decode('utf-8')
                    
                    os.remove('training.log.zip')
                    return logs
                else:
                    logger.error(f"Failed to download artifact. Status code: {response.status_code}")
        
        logger.warning("No training-logs artifact found")
        return "No training logs found in artifacts"
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
            if globals().get('args') and args.debug: # TODO: Fix Do this properly, maybe subclass `discord.Client` for better argument passing
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
                if attachment.filename == "train.py":
                    # Reply to the original message
                    initial_reply = await message.reply("Found train.py! Starting training process...")
                    
                    # Create a new thread from the reply
                    thread = await initial_reply.create_thread(
                        name=f"Training Job - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        auto_archive_duration=1440  # Archive after 24 hours of inactivity
                    )
                    
                    try:
                        # Download the file content
                        logger.info("Downloading train.py content")
                        script_content = await attachment.read()
                        script_content = script_content.decode('utf-8')
                        logger.info("Successfully read train.py content")
                        
                        # Trigger GitHub Action
                        run_id = await trigger_github_action(script_content)
                        
                        if run_id:
                            logger.info(f"Successfully triggered workflow with run ID: {run_id}")
                            await thread.send(f"GitHub Action triggered successfully! Run ID: {run_id}\nMonitoring progress...")
                            
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
                            logger.error("Failed to trigger GitHub Action")
                            await thread.send("Failed to trigger GitHub Action. Please check the configuration.")
                    
                    except Exception as e:
                        logger.error(f"Error processing request: {str(e)}", exc_info=True)
                        await thread.send(f"Error processing request: {str(e)}")
                    
                    break

            if not any(att.filename == "train.py" for att in message.attachments):
                await message.reply("Please attach a file named 'train.py' to your message.")

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
