# src/models_engine/api/models_manager.py

from threading import Lock

from sentence_transformers import CrossEncoder
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

import src.utils.constants as c
from src.utils.logger import setup_logger


class ModelsManager:
    """Singleton class to manage the lifecycle of models used in the API."""
    
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
        self.logger = setup_logger(c.MODELS_LOG_FILE, logger_name=__name__)

        self.logger.debug("Initializing ModelsManager...")

        try:
            # Load models from configuration
            self.embedding_model_name = c.EMBEDDING_MODEL_NAME
            self.reranker_model_name = c.RERANKER_MODEL_NAME
            self.summarization_model_name = c.SUMMARIZATION_MODEL_NAME

            # Initialize Embedding Model
            self.logger.debug(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name)
            self.logger.info(f"Successfully loaded embedding model: {self.embedding_model_name}")

            # Initialize Reranker Model
            self.logger.debug(f"Loading reranker model: {self.reranker_model_name}")
            self.reranker_model = CrossEncoder(self.reranker_model_name, device="cpu")
            self.logger.info(f"Successfully loaded reranker model: {self.reranker_model_name}")

            # Initialize Summarization Model
            self.logger.debug(f"Loading summarization model: {self.summarization_model_name}")
            self.summarizer_tokenizer = AutoTokenizer.from_pretrained(self.summarization_model_name)
            self.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(self.summarization_model_name)
            self.logger.info(f"Successfully loaded summarization model: {self.summarization_model_name}")

        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise RuntimeError("Failed to initialize models.") from e

    def get_embedding_model(self):
        """Returns the embedding model and tokenizer."""
        return self.embedding_tokenizer, self.embedding_model

    def get_reranker_model(self):
        """Returns the reranker (CrossEncoder) model."""
        return self.reranker_model

    def get_summarization_model(self):
        """Returns the summarization model and tokenizer."""
        return self.summarizer_tokenizer, self.summarizer_model

    async def shutdown(self):
        """Performs cleanup during shutdown."""
        self.logger.info("Shutting down and cleaning up models.")
        self.embedding_model = None
        self.embedding_tokenizer = None
        self.reranker_model = None
        self.summarizer_model = None
        self.summarizer_tokenizer = None
        self.logger.info("Models cleanup completed.")
