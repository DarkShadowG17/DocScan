import os
import yaml
from pathlib import Path
from typing import Optional, Dict
from .utils import get_cli_config_path


class ConfigHandler:
    """Class for handling interactions with config.yml"""

    def __init__(self, file_name: str, override_config: bool = True) -> None:  # noqa: D
        # Check if the proper directory exists, if not create it
        if not os.path.exists(self.config_dir):
            cofig_dir = Path(self.config_dir)
            cofig_dir.mkdir(parents=True)
        self.override_config = override_config
        self.file_name = file_name

    @property
    def config_dir(self) -> str:
        """Retrieve Transform config_dir from $TFD_CONFIG_DIR, default config dir is ~/.transform"""
        return str(get_cli_config_path())

    @property
    def config_file_path(self) -> str:
        """Config file can be found at <config_dir>/<file_name>"""
        return os.path.join(self.config_dir, self.file_name)

    def get_config_value(self, key: str) -> Optional[str]:  # noqa: D
        try:
            config_file = open(self.config_file_path, "r")
        except FileNotFoundError:
            return None

        config = yaml.load(config_file, Loader=yaml.Loader)
        if config and key in config and config[key]:
            return config[key]

        return None

    def set_config_value(self, key: str, value: str) -> None:  # noqa: D
        # If the caller passed a falsey value, assume they'd like to remove the key
        if not self.override_config:
            return  # Don't override config if flagged
        if not value:
            self.pop_config_value(key)
            return

        configs: Dict[str, str] = {}
        if os.path.exists(self.config_file_path):
            with open(self.config_file_path) as f:
                configs = yaml.load(f, Loader=yaml.SafeLoader) or {}

        configs[key] = value

        with open(self.config_file_path, "w") as f:
            yaml.dump(configs, f)

    def pop_config_value(self, key: str) -> Optional[str]:  # noqa: D
        if not self.override_config:
            return None  # Don't override config if flagged
        configs: Dict[str, str] = {}
        if os.path.exists(self.config_file_path):
            with open(self.config_file_path) as f:
                configs = yaml.load(f, Loader=yaml.SafeLoader) or {}

        config_value = configs.pop(key, None)

        with open(self.config_file_path, "w") as f:
            yaml.dump(configs, f)

        return config_value

    def reset(self) -> None:  # noqa: D
        if not self.override_config:
            return None  # Don't override config if flagged

        if os.path.exists(self.config_file_path):
            open(self.config_file_path, "w").close()
