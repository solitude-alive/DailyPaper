import os


def create_directory(path: str) -> None:
    """
    Creates a directory if it does not exist.

    Args:
        path (str): The directory path to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def remove_directory(path: str) -> None:
    """
    Remove a directory if it is empty.

    Args:
        path (str): The directory path to remove.
    """
    # Check if the directory is empty
    if not os.listdir(
        path
    ):  # os.listdir(path) returns a list of entries in the directory
        os.rmdir(path)  # Remove the empty directory
        print(f"Directory '{path}' has been removed.")
