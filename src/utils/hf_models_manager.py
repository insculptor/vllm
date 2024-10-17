####################################################################################
#####                File: src/models/hf_models_manager.py                     #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/25/2024                              #####
#####  Hugging Face Models Manager to Load Models from HF (Optional)           #####
####################################################################################

import os
import sys

sys.path.insert(0, os.getenv('ROOT_DIR'))

from huggingface_hub import login, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.config import ConfigLoader


class HFModelsManager:
    def __init__(self, repo_name: str, model_path: str = None):
        """
        Initializes the Hugging Face Model Manager by connecting to Hugging Face and preparing the model.

        Args:
            repo_name (str): The Hugging Face repository name (e.g., "google/gemma-2-2b-it").
            model_path (str, optional): The local base path where the model will be downloaded. 
                                        Defaults to the path from the YAML configuration.
        """
        # Load the configuration based on the environment using ConfigLoader
        config = ConfigLoader()

        self.repo_name = repo_name
        self.model_base_dir = model_path or config.get('MODELS_BASE_DIR', './models')
        print(f"Saving Model to: {self.model_base_dir}")
        self.token = os.getenv('HUGGINGFACE_TOKEN')

        if not self.token:
            raise ValueError("HUGGINGFACE_TOKEN environment variable is not set.")

        # Connect to Hugging Face using the token
        login(token=self.token)
        print("Connected to Hugging Face")

        # Ensure the model directory is prepared and the model is available locally
        self.model_dir = self.ensure_model_available(self.repo_name, self.model_base_dir)

    def ensure_model_available(self, repo_name: str, local_dir: str) -> str:
        """
        Ensure that the model is available locally under model_path/repo_name.
        If not, download it from the Hugging Face repo.

        Args:
            repo_name (str): The Hugging Face model repository name.
            local_dir (str): The local base directory where the repo directory will be created.

        Returns:
            str: The path to the local model directory (model_path/repo_name).
        """
        try:
            # Define the specific path for the model (model_path/repo_name)
            repo_local_dir = os.path.join(local_dir, repo_name)

            # Create the repo_name directory if it doesn't exist
            if not os.path.exists(repo_local_dir):
                os.makedirs(repo_local_dir)

            # Download the model to the repo_local_dir
            model_path = snapshot_download(repo_id=repo_name, local_dir=repo_local_dir)
            print(f"Model '{repo_name}' is available at {model_path}")
            return model_path
        except Exception as e:
            print(f"Failed to download or locate model: {e}")
            raise

    def initialize_model(self):
        """
        Initialize the model and tokenizer from the local model directory.

        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: The initialized model and tokenizer.
        """
        try:
            model = AutoModelForCausalLM.from_pretrained(self.model_dir)
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            print(f"Model and tokenizer for '{self.repo_name}' initialized successfully.")
            return model, tokenizer
        except Exception as e:
            print(f"Failed to initialize model: {e}")
            raise


if __name__ == "__main__":
    # Use the model defined in the YAML configuration
    config_loader = ConfigLoader()
    repo_name = config_loader.get('models')['CAUSAL_MODEL']
    model_path = config_loader.get('dirs')['MODELS_BASE_DIR']

    # Instantiate the HFModelsManager
    manager = HFModelsManager(repo_name=repo_name,model_path=model_path)

    # Initialize the model and tokenizer
    model, tokenizer = manager.initialize_model()
