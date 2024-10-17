# src/models_engine/api/models_manager.py

from threading import Lock

from transformers import AutoModel, AutoTokenizer

from src.utils.config import ConfigLoader
from src.utils.logger import setup_logger


class ModelsManager:
    """
    Singleton class to manage the lifecycle of models used in the API.
    """
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        """Create a singleton instance."""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls, *args, **kwargs)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize models and logger."""
        self.config = ConfigLoader()
        log_file_name = self.config.get("logger")["models_log_file"]
        self.logger = setup_logger(log_file_name, logger_name=__name__)

        self.logger.debug("Initializing ModelsManager...")
        try:
            # Load models from config
            self.embedding_model_name = self.config.get("models")["EMBEDDING_MODEL"]
            self.reranker_model_name = self.config.get("models")["RERANKER_MODEL"]

            # Load embedding model
            self.logger.debug(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name)
            self.logger.info(f"Successfully loaded embedding model: {self.embedding_model_name}")

            # Load reranker model
            self.logger.debug(f"Loading reranker model: {self.reranker_model_name}")
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
            self.reranker_model = AutoModel.from_pretrained(self.reranker_model_name)
            self.logger.info(f"Successfully loaded reranker model: {self.reranker_model_name}")

        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise RuntimeError("Failed to initialize models.") from e

    def get_embedding_model(self):
        """Returns the embedding model and tokenizer."""
        return self.embedding_tokenizer, self.embedding_model

    def get_reranker_model(self):
        """Returns the reranker model and tokenizer."""
        return self.reranker_tokenizer, self.reranker_model

    async def shutdown(self):
        """Performs cleanup during shutdown."""
        self.logger.info("Shutting down and cleaning up models.")
        self.embedding_model = None
        self.embedding_tokenizer = None
        self.reranker_model = None
        self.reranker_tokenizer = None
        self.logger.info("Models cleanup completed.")
