# src/utils/config.py

import os

import yaml


class ConfigLoader:
    def __init__(self):
        # Get environment (defaults to 'sit')
        self.env = os.getenv("ENV", "local")
        print(f"[INFO] Running in Environment: {self.env}")
        # Use the current script directory to resolve the project root path
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

        # Construct the absolute path to the config file
        self.config_file = os.path.join(base_dir, 'config', f'{self.env}.yaml')
        print(f"Loading configuration from: {self.config_file}")

        # Load the configuration from the YAML file
        self.config = self._load_config()

    def _load_config(self):
        try:
            with open(self.config_file, "r") as file:
                config = yaml.safe_load(file)
                return config
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Configuration file {self.config_file} not found. Please check the file path.") from e

    def get(self, key_path, default=None):
        keys = key_path.split('.')
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def __getitem__(self, key):
        return self.get(key)

    def __contains__(self, key):
        return self.get(key) is not None
