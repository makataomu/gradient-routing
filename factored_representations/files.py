import io
import os
import zipfile
from pathlib import Path

import requests

SHARED_FILES_PATH = os.path.expanduser("~/team-shard-filesystem/")


def download_unzip_and_locate(
    url: str, password: str, output_repo_dir: str, filename: str
) -> str:
    """
    Downloads a zip file from a given URL, unzips it entirely using the provided
    password while retaining the original directory structure, and locates a specific
    file within the extracted contents.
    If the zip file has already been extracted, it will search for the specified file
    within the existing contents.

    Args:
        url: The URL of the zip file to download.
        password: The password to use for extracting the zip file.
        output_repo_dir: The directory to store the extracted contents. File path should
            be relative to the repository root.
        filename: The name of the file to locate within the extracted contents.

    Returns:
        The full path to the located file.

    Raises:
        requests.HTTPError: If there's an HTTP error while downloading the file.
        zipfile.BadZipFile: If the zip file is corrupted.
        RuntimeError: If the provided password is incorrect.

    Example:
        >>> url = "https://example.com/data.zip"
        >>> password = "secret"
        >>> output_dir = "path/to/your/output/directory"
        >>> filename = "important_data.txt"
        >>> result = download_unzip_and_locate(url, password, output_dir, filename)
        >>> if result:
        ...     print(f"File found at: {result}")
        ... else:
        ...     print("File not found")
    """
    # Create the output directory if it doesn't exist
    output_dir = ensure_repo_dir_exists(output_repo_dir)

    print(f"Searching for file {filename} in {output_dir}...")
    # Check if the file already exists in the output directory or any subdirectory
    for root, _, files in os.walk(output_dir):
        if filename in files:
            full_path = os.path.join(root, filename)
            print(f"File {filename} found at: {full_path}")
            return full_path

    # If the file doesn't exist, download and extract the zip file
    print(f"Downloading file from {url}...")
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Extract the contents of the zip file
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        # Extract all files while retaining the directory structure
        zf.extractall(path=output_dir, pwd=password.encode())
        print("Zip file extracted successfully.")

    # Search for the specific file in the extracted contents
    for root, _, files in os.walk(output_dir):
        if filename in files:
            full_path = os.path.join(root, filename)
            print(f"File {filename} found at: {full_path}")
            return full_path

    # If the file is not found after extraction, raise an error
    raise RuntimeError(f"File {filename} not found in the extracted contents.")


def repo_path_to_abs_path(path: str) -> Path:
    """
    Convert a path relative to the repository root to an absolute path.

    Args:
        path: A path relative to the repository root.

    Returns:
        The absolute path.
    """
    repo_abs_path = Path(__file__).parent.parent.absolute()
    return repo_abs_path / path


def ensure_repo_dir_exists(repo_path: str) -> Path:
    """
    Ensure that a directory exists within the repository root.

    Args:
        repo_path: A path relative to the repository root.

    Returns:
        The absolute path to the directory.
    """
    cache_directory = repo_path_to_abs_path(repo_path)
    cache_directory.mkdir(parents=True, exist_ok=True)
    return cache_directory


def ensure_shared_dir_exists(path: str | Path) -> Path:
    """
    Get the full path to a file or directory within the shared filesystem.

    Args:
        path: A path relative to the shared filesystem root.

    Returns:
        The full path to the file or directory.
    """
    assert os.path.exists(SHARED_FILES_PATH), f"Path {SHARED_FILES_PATH} doesn't exist."
    shared_dir = Path(SHARED_FILES_PATH) / path
    shared_dir.mkdir(parents=True, exist_ok=True)
    return shared_dir
