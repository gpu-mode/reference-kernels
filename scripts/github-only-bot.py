"""
This is a legacy file and no longer needed but keeping it a round since its simplest example I have of triggering a Github action using a python script
"""

from dotenv import load_dotenv
from github import Github
import os
import time
from datetime import datetime, timezone
import requests

# Load environment variables
load_dotenv()

def trigger_github_action():
    """
    Triggers the GitHub action and returns the latest run ID
    """
    gh = Github(os.getenv('GITHUB_TOKEN'))
    repo = gh.get_repo(os.getenv('GITHUB_REPO'))
    
    try:
        # Record the time before triggering
        trigger_time = datetime.now(timezone.utc)
        
        # Trigger the workflow
        workflow = repo.get_workflow("train_workflow.yml")
        success = workflow.create_dispatch("main")
        
        if success:
            # Wait a moment for the run to be created
            time.sleep(2)
            
            # Get runs created after our trigger time
            runs = list(workflow.get_runs())
            for run in runs:
                if run.created_at.replace(tzinfo=timezone.utc) > trigger_time:
                    return run.id
            
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def download_artifact(run_id):
    """
    Downloads the training log artifact from the workflow run
    """
    gh = Github(os.getenv('GITHUB_TOKEN'))
    repo = gh.get_repo(os.getenv('GITHUB_REPO'))
    
    # Get the specific run
    run = repo.get_workflow_run(run_id)
    
    # Get artifacts from the run
    artifacts = run.get_artifacts()
    
    for artifact in artifacts:
        if artifact.name == 'training-logs':
            # Download the artifact
            url = artifact.archive_download_url
            headers = {'Authorization': f'token {os.getenv("GITHUB_TOKEN")}'}
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                with open('training.log.zip', 'wb') as f:
                    f.write(response.content)
                
                # Read the log file from the zip
                import zipfile
                with zipfile.ZipFile('training.log.zip') as z:
                    with z.open('training.log') as f:
                        logs = f.read().decode('utf-8')
                
                # Clean up the zip file
                os.remove('training.log.zip')
                return logs
    
    return "No training logs found in artifacts"

def check_workflow_status(run_id):
    """
    Monitors the GitHub Action workflow status
    """
    gh = Github(os.getenv('GITHUB_TOKEN'))
    repo = gh.get_repo(os.getenv('GITHUB_REPO'))
    
    while True:
        run = repo.get_workflow_run(run_id)
        
        if run.status == "completed":
            logs = download_artifact(run_id)
            return run.conclusion, logs, run.html_url
        
        print(f"Workflow still running... Status: {run.status}")
        print(f"Live view: {run.html_url}")
        time.sleep(30)

if __name__ == "__main__":
    run_id = trigger_github_action()
    
    if run_id:
        print(f"GitHub Action triggered successfully! Run ID: {run_id}")
        print("Monitoring progress...")
        
        # Monitor the workflow
        status, logs, url = check_workflow_status(run_id)
        
        print(f"\nWorkflow completed with status: {status}")
        print("\nTraining Logs:")
        print(logs)
        print(f"\nView the full run at: {url}")
    else:
        print("Failed to trigger GitHub Action. Please check your configuration.")