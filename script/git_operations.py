import subprocess


def git_commit_and_push(date=None, base_path="summaries"):
    """
    Automates git commit and push process.
    """
    assert date is not None, "Date cannot be None"
    branch = "daily-updates-{}".format(date)
    attempt = 0
    retries = 3
    subprocess.run(["git", "checkout", "-b", branch], check=True)
    while attempt < retries:
        try:
            subprocess.run(["git", "add", "*.md"], check=True)
            subprocess.run(["git", "add", "summaries/*.md"], check=True)
            subprocess.run(["git", "add", "summaries/*/*.md"], check=True)
            subprocess.run(["git", "commit", "-m", "Daily update"], check=True)
            subprocess.run(["git", "push", "-u", "origin", branch], check=True)
            break
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while committing and pushing: {e}")
            attempt += 1
            if attempt < retries:
                print(f"Retrying... (Attempt {attempt}/{retries})")
            else:
                print("Max retries reached. Exiting.")


def create_pull_request(date=None):
    """
    Automates pull request creation using GitHub CLI.
    """
    assert date is not None, "Date cannot be None"
    branch = "daily-updates-{}".format(date)
    try:
        subprocess.run(
            [
                "gh",
                "pr",
                "create",
                "--base",
                "main",
                "--head",
                branch,
                "--title",
                "Daily Updates",
                "--body",
                "Auto-generated updates",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while creating PR: {e}")
