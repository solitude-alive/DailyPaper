import subprocess


def git_commit_and_push(date: str = None) -> None:
    """
    Automates git commit and push process.

    Args:
        date (str): The date of the papers.

    Raises:
        AssertionError: If an error occurs while committing and pushing.
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
                raise AssertionError(
                    f"Error occurred while committing and pushing {e}"
                ) from e


def create_pull_request(date: str = None) -> None:
    """
    Automates pull request creation using GitHub CLI.

    Args:
        date (str): The date of the papers.

    Raises:
        AssertionError: If an error occurs while creating a pull request.
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
        raise AssertionError(f"Error occurred while creating PR: {e}") from e
