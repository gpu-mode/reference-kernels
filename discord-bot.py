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
        # Record the time before triggering
        trigger_time = datetime.now(timezone.utc)
        
        # Log workflow attempt
        logger.info(f"Looking for workflow 'train_workflow.yml' in repo {os.getenv('GITHUB_REPO')}")
        
        # Trigger the workflow with the script content
        workflow = repo.get_workflow("train_workflow.yml")
        logger.info("Found workflow, attempting to dispatch")
        
        success = workflow.create_dispatch("main", {'script_content': script_content})
        logger.info(f"Workflow dispatch result: {success}")
        
        if success:
            # Wait a moment for the run to be created
            await asyncio.sleep(2)
            
            # Get runs created after our trigger time
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
        # Get the specific run
        run = repo.get_workflow_run(run_id)
        
        # Get artifacts from the run
        artifacts = run.get_artifacts()
        logger.info(f"Found {artifacts.totalCount} artifacts")
        
        for artifact in artifacts:
            logger.info(f"Found artifact: {artifact.name}")
            if artifact.name == 'training-logs':
                # Download the artifact
                url = artifact.archive_download_url
                headers = {'Authorization': f'token {os.getenv("GITHUB_TOKEN")}'}
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    logger.info("Successfully downloaded artifact")
                    with open('training.log.zip', 'wb') as f:
                        f.write(response.content)
                    
                    # Read the log file from the zip
                    with zipfile.ZipFile('training.log.zip') as z:
                        with z.open('training.log') as f:
                            logs = f.read().decode('utf-8')
                    
                    # Clean up the zip file
                    os.remove('training.log.zip')
                    return logs
                else:
                    logger.error(f"Failed to download artifact. Status code: {response.status_code}")
        
        logger.warning("No training-logs artifact found")
        return "No training logs found in artifacts"
    except Exception as e:
        logger.error(f"Error in download_artifact: {str(e)}", exc_info=True)
        return f"Error downloading artifacts: {str(e)}"

async def check_workflow_status(run_id, message):
    """
    Monitors the GitHub Action workflow status and updates Discord
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
            
            await message.channel.send(f"Workflow still running... Status: {run.status}\nLive view: {run.html_url}")
            await asyncio.sleep(30)
        except Exception as e:
            logger.error(f"Error in check_workflow_status: {str(e)}", exc_info=True)
            return "error", str(e), None

@client.event
async def on_ready():
    logger.info(f'Logged in as {client.user}')

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
                    await message.channel.send("Found train.py! Starting training process...")
                    
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
                            await message.channel.send(f"GitHub Action triggered successfully! Run ID: {run_id}\nMonitoring progress...")
                            
                            # Monitor the workflow
                            status, logs, url = await check_workflow_status(run_id, message)
                            
                            # Send results back to Discord
                            await message.channel.send(f"Training completed with status: {status}")
                            
                            # Split logs if they're too long for Discord's message limit
                            if len(logs) > 1900:
                                chunks = [logs[i:i+1900] for i in range(0, len(logs), 1900)]
                                for i, chunk in enumerate(chunks):
                                    await message.channel.send(f"```\nLogs (part {i+1}/{len(chunks)}):\n{chunk}\n```")
                            else:
                                await message.channel.send(f"```\nLogs:\n{logs}\n```")
                            
                            if url:
                                await message.channel.send(f"View the full run at: {url}")
                        else:
                            logger.error("Failed to trigger GitHub Action")
                            await message.channel.send("Failed to trigger GitHub Action. Please check the configuration.")
                    
                    except Exception as e:
                        logger.error(f"Error processing request: {str(e)}", exc_info=True)
                        await message.channel.send(f"Error processing request: {str(e)}")
                    
                    break

            if not any(att.filename == "train.py" for att in message.attachments):
                await message.channel.send("Please attach a file named 'train.py' to your message.")

# Run the bot
if __name__ == "__main__":
    logger.info("Starting bot...")
    client.run(os.getenv('DISCORD_TOKEN'))