from dotenv import load_dotenv
from github import Github
import os
import time

# Load environment variables
load_dotenv()

def trigger_github_action(file_path='train.py'):
    """
    Triggers a GitHub action after updating train.py
    Returns the run ID for monitoring
    """
    # Initialize GitHub client
    gh = Github(os.getenv('GITHUB_TOKEN'))
    repo = gh.get_repo(os.getenv('GITHUB_REPO'))
    
    try:
        # Read the local train.py file
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Update or create train.py in the repository
        try:
            file = repo.get_contents("train.py")
            repo.update_file(
                "train.py",
                "Update train.py via script",
                content,
                file.sha
            )
        except Exception:
            repo.create_file(
                "train.py",
                "Create train.py via script",
                content
            )
        
        # Trigger the workflow
        workflow = repo.get_workflow("train_workflow.yml")
        run = workflow.create_dispatch("main")
        return run.id
    
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
    # Trigger the action
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