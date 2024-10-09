# /src/vdb_engine/engine/embedding.py

from pathlib import Path

import torch
from angle_emb import AnglE
from transformers import pipeline

from src.utils.config import ConfigLoader
from src.utils.logger import setup_logger


class EmbeddingEngine:
    """Class for generating embeddings and summaries."""

    def __init__(self):
        # Initialize logger and config
        self.logger = setup_logger(filename=__name__)
        self.config = ConfigLoader()

        # Get configurations
        self.models_config = self.config.get('models', {})
        self.dirs_config = self.config.get('dirs', {})

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        # Model paths
        self.models_base_dir = Path(self.dirs_config.get('MODELS_BASE_DIR', './models'))
        self.embedding_model_name = self.models_config.get('EMBEDDING_MODEL', 'your_embedding_model_name')
        self.summarization_model_name = self.models_config.get('SUMMARIZATION_MODEL', 'your_summarization_model_name')

        # Load models
        self.embedding_model = self.load_embedding_model()
        self.summarizer = self.load_summarizer()

    def load_embedding_model(self):
        try:
            model_path = self.models_base_dir / self.embedding_model_name / 'model'
            if not model_path.exists():
                # Download and save the model
                self.logger.info(f"Downloading embedding model: {self.embedding_model_name}")
                angle = pipeline("feature-extraction", model=self.embedding_model_name)
                angle.model.save_pretrained(model_path)
                angle.tokenizer.save_pretrained(model_path)
                self.logger.info(f"Downloaded and saved embedding model to {model_path}")
            else:
                self.logger.info(f"Loading embedding model from {model_path}")

            angle = AnglE.from_pretrained(model_name_or_path=str(model_path), pooling_strategy='cls').to(self.device)
            self.logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            return angle
        except Exception as e:
            self.logger.exception(f"Failed to load embedding model: {e}")
            raise

    def load_summarizer(self):
        try:
            model_path = self.models_base_dir / self.summarization_model_name / 'model'
            if not model_path.exists():
                # Download and save the model
                self.logger.info(f"Downloading summarization model: {self.summarization_model_name}")
                summarizer = pipeline("summarization", model=self.summarization_model_name)
                summarizer.model.save_pretrained(model_path)
                summarizer.tokenizer.save_pretrained(model_path)
                self.logger.info(f"Downloaded and saved summarization model to {model_path}")
            else:
                self.logger.info(f"Loading summarization model from {model_path}")

            summarizer = pipeline("summarization", model=str(model_path), device=0 if self.device.type == 'cuda' else -1)
            self.logger.info(f"Loaded summarization model: {self.summarization_model_name}")
            return summarizer
        except Exception as e:
            self.logger.exception(f"Failed to load summarization model: {e}")
            raise

    def summarize_text(self, text):
        try:
            max_length = max(100, int(len(text.split()) / 2))
            self.logger.debug(f"Summarizing text with max_length={max_length}")
            summary = self.summarizer(text[:1024], max_length=max_length, min_length=100, do_sample=False)
            summary_text = summary[0]['summary_text']
            self.logger.info("Generated summary")
            return summary_text
        except Exception as e:
            self.logger.exception(f"Failed to summarize text: {e}")
            raise

    def generate_embedding(self, text):
        try:
            embedding = self.embedding_model.encode(text, to_numpy=True)
            self.logger.info("Generated embedding")
            return embedding[0]  # Assuming we get a list of embeddings
        except Exception as e:
            self.logger.exception(f"Failed to generate embedding: {e}")
            raise

    def process_document(self, document_text):
        """Process a single document: summarize and generate embedding."""
        try:
            self.logger.info("Processing document")
            summary = self.summarize_text(document_text)
            embedding = self.generate_embedding(summary)
            self.logger.info("Processed document successfully")
            return summary, embedding
        except Exception as e:
            self.logger.exception(f"Failed to process document: {e}")
            raise

    # Additional methods for handling chunks can be added here
