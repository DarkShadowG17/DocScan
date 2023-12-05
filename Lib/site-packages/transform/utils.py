import os
import time
from pathlib import Path

import git
import hashlib

from typing import Tuple, Optional

LOCAL = "local"


def is_git_repo(path: str = None) -> bool:
    """Check is a path is a git repo"""
    try:
        git.Repo(path, search_parent_directories=True)
        return True
    except git.exc.InvalidGitRepositoryError:
        return False
    return False


def get_git_info(path: str = None) -> Tuple[str, str, str]:
    """Attempts to retrieve git repo info"""
    repo = git.Repo(path, search_parent_directories=True)  # will throw an error if this is not a git repo

    # NOTE: we are taking the name of the directory as the name of the repo
    # TODO: better handling of detached state
    repo_name = os.path.split(os.path.realpath(os.path.dirname(repo.common_dir)))[-1]
    branch = repo.active_branch.name if not repo.head.is_detached else f"detached_{repo.head.object.hexsha}"
    commit = repo.head.object.hexsha

    return repo_name, branch, commit


def local_commit_info() -> Tuple[str, str, str]:
    """Create a pseudo-git setup for validation from a non-git repo"""
    hasher = hashlib.sha1()
    hasher.update(str(time.time()).encode("utf-8"))
    commit = str(hasher.hexdigest())
    return LOCAL, LOCAL, commit


def resolve_local_commit_info(config_dir: str) -> Tuple[str, str, str]:
    """Includes logic for uploading Transform models via the CLI."""
    if is_git_repo(config_dir):
        repo, branch, commit = get_git_info(config_dir)
    else:
        repo, branch, commit = local_commit_info()

    # to differentiate between CI-based commits and local CLI-based commits we edit the commit sha
    commit += f"-dirty-{int(time.time())}"
    return repo, branch, commit


def directory_last_edited(config_dir: str) -> Optional[float]:
    """Returns the last time a file or folder was edited in the given directory. None if folder doesn't exist."""
    if not os.path.exists(config_dir):
        return None
    last_edited = os.path.getmtime(config_dir)
    for path, folders, filenames in os.walk(config_dir):
        folders[:] = [folder for folder in folders if not folder.startswith(".")]
        for filename in filenames:
            if filename.endswith(".yml") or filename.endswith(".yaml"):
                last_edited = max(last_edited, os.path.getmtime(os.path.join(path, filename)))
        for folder in folders:
            last_edited = max(last_edited, os.path.getmtime(os.path.join(path, folder)))
    return last_edited


def get_cli_config_path() -> Path:
    """Returns the path of the transform config directory."""
    config_dir = os.getenv("TFD_CONFIG_DIR")
    return Path(config_dir).resolve().absolute() if config_dir else Path.home() / ".transform"
