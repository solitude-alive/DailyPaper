import subprocess

def git_commit_and_push(date=None):
    """
    Automates git commit and push process.
    """
    assert date is not None, "Date cannot be None"
    branch = "daily-updates-{}".format(date)
    subprocess.run(["git", "checkout", "-b", branch], check=True)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "Daily update"], check=True)
    subprocess.run(["git", "push", "-u", "origin", branch], check=True)

def create_pull_request(date=None):
    """
    Automates pull request creation using GitHub CLI.
    """
    assert date is not None, "Date cannot be None"
    branch = "daily-updates-{}".format(date)
    try:
        subprocess.run(["gh", "pr", "create", "--base", "main", "--head", branch, "--title", "Daily Updates", "--body",
                        "Auto-generated updates"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while creating PR: {e}")
