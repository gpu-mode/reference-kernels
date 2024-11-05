from dotenv import load_dotenv
from github import Github
import os
import time

# Load environment variables
load_dotenv()

def trigger_github_action():
    """
    Triggers the GitHub action and returns the latest run ID
    """
    gh = Github(os.getenv('GITHUB_TOKEN'))
    repo = gh.get_repo(os.getenv('GITHUB_REPO'))
    
    try:
        # Trigger the workflow
        workflow = repo.get_workflow("train_workflow.yml")
        success = workflow.create_dispatch("main")
        
        if success:
            # Get the latest run ID
            runs = list(workflow.get_runs())
            if runs:
                return runs[0].id  # Get the most recent run
            
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def check_workflow_status(run_id):
    """
    Monitors the GitHub Action workflow status
    Returns the conclusion and logs when complete
    """
    gh = Github(os.getenv('GITHUB_TOKEN'))
    repo = gh.get_repo(os.getenv('GITHUB_REPO'))
    
    while True:
        runs = repo.get_workflow_runs()
        for run in runs:
            if run.id == run_id:
                if run.status == "completed":
                    return run.conclusion, run.html_url
        print("Workflow still running... checking again in 30 seconds")
        time.sleep(30)

if __name__ == "__main__":
    run_id = trigger_github_action()
    
    if run_id:
        print(f"GitHub Action triggered successfully! Run ID: {run_id}")
        print("Monitoring progress...")
        
        # Monitor the workflow
        status, url = check_workflow_status(run_id)
        
        print(f"\nWorkflow completed with status: {status}")
        print(f"View the full run at: {url}")
    else:
        print("Failed to trigger GitHub Action. Please check your configuration.")