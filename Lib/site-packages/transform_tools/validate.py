import logging
import json
import os
import requests
import sys
import yaml
import git
import xxhash
from typing import Optional, Dict

EXPECTED_TF_CONFIG_FILE_NAMES = ["tdfconfig.yml", "tdfconfig.yaml", "validate_configs.yaml", "commit_configs.yaml"]
TRANSFORM_API_URL = "https://api.transformdata.io"
UPLOAD_MODE_VALIDATE = "validate"
UPLOAD_MODE_COMMIT = "commit"
LOCAL_DIR_DEFAULT = "."
# TODO: Return error response so this constant doesn't have to be passed to the CLI
ERROR_RESPONSE_PREFIX = "Error response: "

logger = logging.getLogger(__name__)


def _err_msg_from_err_response(r: requests.Response) -> str:
    # Typically I'm against exceptions for control flow, but meh it's readable
    # in this case
    try:
        error_dict = json.loads(r.text)["error"]
        err_msg = f"{error_dict['error_type']}: {error_dict['message']}"
    except:  # noqa: E722
        err_msg = r.text

    return err_msg


def read_config_files(config_dir: str) -> Dict:  # noqa: D
    try:
        repo = git.Repo(".", search_parent_directories=True)
        git_dir = repo.working_tree_dir
    except git.exc.InvalidGitRepositoryError:
        git_dir = None

    return _read_config_files(config_dir, git_dir)


def _read_config_files(config_dir: str, git_dir: Optional[str]) -> Dict:
    """Read yaml files from config_dir. Returns (file name, file contents) per file in dir"""
    assert os.path.exists(config_dir), f"User-specified config dir ({config_dir}) does not exist"

    results = {}
    for path, _folders, filenames in os.walk(config_dir):
        # This ensures we skip files in the dir's .git
        # For reference we ran into an issue wherein a remote branch
        # was named such that in ended in `.yaml`. And the branch metadata
        # file in .git/logs/refs/remotes/origins was getting uploaded
        # and causing parsing errors
        if "/.git/" in path:
            continue

        for fname in filenames:
            if not (fname.endswith(".yml") or fname.endswith(".yaml")):
                continue

            # ignore transform config
            if fname in EXPECTED_TF_CONFIG_FILE_NAMES:
                continue

            with open(os.path.join(path, fname), "r") as f:
                filepath = os.path.abspath(os.path.join(path, fname))
                if git_dir is not None:
                    filepath = filepath.split(git_dir)[-1]
                results[filepath] = f.read()
                try:
                    yaml.safe_load_all(results[filepath])
                except yaml.parser.ParserError as e:
                    raise yaml.parser.ParserError(f"Invalid yaml in config file at path: {filepath}. {e}")
                except Exception as e:
                    raise Exception(f"Failed loading yaml config file. {e}")

    return results


def hash_file(path: str, content: str) -> str:  # noqa: D
    file_hash = xxhash.xxh3_128()
    file_hash.update(path.encode())
    file_hash.update(content.encode())
    return file_hash.hexdigest()


def hash_config_files(yaml_files: Dict[str, str]) -> Dict[str, str]:  # noqa: D
    hash_to_filename = {}
    for filename, contents in yaml_files.items():
        hashed_file = hash_file(filename, contents)
        hash_to_filename[hashed_file] = filename
    return hash_to_filename


def commit_configs(
    auth_header: Dict[str, str],
    repo: str,
    branch: str,
    commit: str,
    config_dir: str = LOCAL_DIR_DEFAULT,  # default to local dir
    is_validation: bool = False,
    api_url: str = TRANSFORM_API_URL,  # default to prod api
) -> requests.Response:
    """Creates either a validated or validation model based on `is_validation`

    Parses configs, runs semantic validations, and creates a validation or validated
    model based on `is_validation`
    """
    # Sometimes people accidentally override their local TRANSFORM_API_URL
    # environment variable with an empty string instead of UNSET-ing it.
    # And then an empty string propagates all the way here. So if api_url
    # is empty, default it. It's worth noting that in python "" evaluates to
    # false, which is why this oneliner works.
    api_url = api_url or TRANSFORM_API_URL

    # Clean up branch name because people like to put slashes in their branch names
    if "/" in branch:
        branch = branch.replace("/", "__")  # dunder, for readability... to the extent it matters

    # get the config files
    yaml_files = read_config_files(config_dir)
    hash_to_filename = hash_config_files(yaml_files)
    yaml_hashes = list(hash_to_filename.keys())

    compare_hashes_body = {"hashes": yaml_hashes}

    headers = {**{"Content-Type": "application/json"}, **auth_header}
    verify = api_url.startswith("https")

    compare_hashes_url = f"{api_url}/api/v1/model/{repo}/{branch}/{commit}/compare_yaml_hashes"
    logger.info("Uploading config file hashes")
    r = requests.post(
        compare_hashes_url, data=json.dumps(compare_hashes_body).encode("utf-8"), headers=headers, verify=verify
    )
    if r.status_code != 200:
        err_msg = _err_msg_from_err_response(r)
        raise Exception(err_msg)

    results = r.json()

    unmatched_hashes = yaml_hashes
    if results["unmatched"] is not None:
        unmatched_hashes = results["unmatched"]

    matched_hashes = results["matched"]

    upload_files = {}
    for yaml_hash in unmatched_hashes:
        yaml_file = hash_to_filename[yaml_hash]
        upload_files[yaml_file] = yaml_files[yaml_file]

    logger.info(f"Files to upload: {upload_files.keys()}")

    add_model_files_body = {
        "yaml_files": upload_files,
        "yaml_hashes": matched_hashes,
    }

    # Add the config files to backend file storage
    add_files_url = f"{api_url}/api/v1/model/{repo}/{branch}/{commit}/add_model_files"
    logger.info(f"add_files_url: {add_files_url}")
    logger.info("Uploading config files")
    r = requests.post(
        add_files_url, data=json.dumps(add_model_files_body).encode("utf-8"), headers=headers, verify=verify
    )
    if r.status_code != 200:
        raise Exception(f"Failed uploading config yaml files. {r.text}")

    # This route validates the configs for files we just pushed and, if there are no
    # errors it then creates a semantic model from those configs in the backend database
    commit_url = f"{api_url}/api/v1/model/{repo}/{branch}/{commit}/commit_model"
    logger.info(f"commit_url: {commit_url}")
    logger.info("Committing model")
    r = requests.post(
        commit_url,
        headers=headers,
        verify=verify,
        json=json.dumps({"is_current": False, "is_validation": is_validation}),
    )
    if r.status_code != 200:
        err_msg = _err_msg_from_err_response(r)
        raise Exception(err_msg)
    logger.info("Successfully committed configs")

    return r


def commit_configs_as_primary(
    auth_header: Dict[str, str],
    repo: str,
    branch: str,
    commit: str,
    config_dir: str = LOCAL_DIR_DEFAULT,  # default to local dir
    api_url: str = TRANSFORM_API_URL,  # default to prod api
) -> requests.Response:
    """Creates a model from the configs and makes it the primary model

    Parses configs, runs semantic validations, and creates a validated model configs
    which is immediately made primary for the organization
    """
    response = commit_configs(
        auth_header=auth_header,
        repo=repo,
        branch=branch,
        commit=commit,
        config_dir=config_dir,
        is_validation=False,
        api_url=api_url,
    )

    promote_model(
        auth_header=auth_header,
        repo=repo,
        branch=branch,
        commit=commit,
        api_url=api_url,
    )

    return response


def validate_configs(
    auth_header: Dict[str, str],
    repo: str,
    branch: str,
    commit: str,
    config_dir: str = LOCAL_DIR_DEFAULT,  # default to local dir
    api_url: str = TRANSFORM_API_URL,  # default to prod api
) -> requests.Response:
    """Parses configs, runs semantic validations, and creates a validation model"""
    return commit_configs(
        auth_header=auth_header,
        repo=repo,
        branch=branch,
        commit=commit,
        config_dir=config_dir,
        is_validation=True,
        api_url=api_url,
    )


def promote_model(
    auth_header: Dict[str, str],
    repo: str,
    branch: str,
    commit: str,
    api_url: str = TRANSFORM_API_URL,  # default to prod api
) -> requests.Response:
    """Promotes an existing model to be the primary model for an organization"""
    verify = api_url.startswith("https")
    promote_url = f"{api_url}/api/v1/model/{repo}/{branch}/{commit}/promote"

    logger.info(f"promote_url: {promote_url}")
    logger.info("Promoting model")
    r = requests.post(promote_url, headers=auth_header, verify=verify)
    if r.status_code != 200:
        err_msg = _err_msg_from_err_response(r)
        raise Exception(err_msg)

    logger.info("Successfully promoted model")
    return r


if __name__ == "__main__":
    mode = sys.argv[1]
    IS_CURRENT = None
    IS_VALIDATION = False
    if mode != UPLOAD_MODE_VALIDATE and mode != UPLOAD_MODE_COMMIT:
        raise ValueError(f"Invalid upload mode ({mode}) passed via args.")

    # Retrieve git info and API key from env
    REPO = os.getenv("REPO")
    # remove github org from repo
    if REPO:
        REPO = "/".join(REPO.split("/")[1:])

    BRANCH: Optional[str]
    if os.getenv("GITHUB_HEAD_REF") == "":
        github_ref = os.getenv("GITHUB_REF")
        if github_ref:
            BRANCH = github_ref.lstrip("/refs/heads/")
    else:
        BRANCH = os.getenv("GITHUB_HEAD_REF")

    API_URL = os.getenv("TRANSFORM_API_URL", TRANSFORM_API_URL)
    COMMIT = os.getenv("GITHUB_SHA")
    TRANSFORM_CONFIG_DIR = os.getenv("TRANSFORM_CONFIG_DIR")
    if TRANSFORM_CONFIG_DIR:
        TRANFSFORM_CONFIG_DIR = TRANSFORM_CONFIG_DIR.lstrip().rstrip()
    TRANSFORM_API_KEY = os.environ["TRANSFORM_API_KEY"].lstrip().rstrip()  # fail if TRANSFORM_API_KEY not present
    auth_header = {"Authorization": f"X-Api-Key {TRANSFORM_API_KEY}"}

    # This protects againsts the case where the varaible is None, making it "None"
    # In practice we've only seen this in our own org, and way back in March of 2021,
    # but I'd rather be safe than sorry
    REPO = f"{REPO}"
    BRANCH = f"{BRANCH}"
    COMMIT = f"{COMMIT}"

    try:
        if mode == UPLOAD_MODE_VALIDATE:
            validate_configs(
                auth_header=auth_header,
                repo=REPO,
                branch=BRANCH,
                commit=COMMIT,
                config_dir=TRANSFORM_CONFIG_DIR or LOCAL_DIR_DEFAULT,
                api_url=API_URL,
            )
        else:  # mode == UPLOAD_MODE_COMMIT
            commit_configs_as_primary(
                auth_header=auth_header,
                repo=REPO,
                branch=BRANCH,
                commit=COMMIT,
                config_dir=TRANSFORM_CONFIG_DIR or LOCAL_DIR_DEFAULT,
                api_url=API_URL,
            )
    except Exception as e:
        print(e)
        sys.exit(1)

    print("success")
