# /src/vdb_engine/engine/retrieval.py

from pathlib import Path

import faiss
import numpy as np

from src.utils.config import ConfigLoader
from src.utils.logger import setup_logger


class VectorDBManager:
    """Base class for managing VectorDB operations."""

    def __init__(self):
        # Initialize logger and config
        self.logger = setup_logger(filename=__name__)
        self.config = ConfigLoader()

        # Get configurations
        self.vectordb_config = self.config.get('vectordb', {})
        self.dirs_config = self.config.get('dirs', {})

        # VectorDB configurations
        self.vectorstore_base_path = Path(self.vectordb_config.get('VECTORSTORE_BASE_DIR', './vectorstore'))
        self.dim = self.vectordb_config.get('EMBEDDING_DIM', 1024)
        self.l2_index = None
        self.hnsw_index = None

        self.logger.info(f"VectorDBManager initialized with base path: {self.vectorstore_base_path}")

    def load_l2_index(self):
        try:
            index_path = self.vectorstore_base_path / self.vectordb_config.get('FAISS_L2_INDEX', 'faiss_l2.index')
            if index_path.exists():
                self.l2_index = faiss.read_index(str(index_path))
                self.logger.info(f"Loaded L2 index from {index_path}")
            else:
                self.l2_index = faiss.IndexFlatL2(self.dim)
                self.logger.info("Created new L2 index")
            return self.l2_index
        except Exception as e:
            self.logger.exception(f"Failed to load L2 index: {e}")
            raise

    def load_hnsw_index(self):
        try:
            index_path = self.vectorstore_base_path / self.vectordb_config.get('FAISS_HNSW_INDEX', 'faiss_hnsw.index')
            if index_path.exists():
                self.hnsw_index = faiss.read_index(str(index_path))
                self.logger.info(f"Loaded HNSW index from {index_path}")
            else:
                M = self.vectordb_config.get('HNSW_M', 32)
                efConstruction = self.vectordb_config.get('HNSW_EF_CONSTRUCTION', 200)
                self.hnsw_index = faiss.IndexHNSWFlat(self.dim, M)
                self.hnsw_index.hnsw.efConstruction = efConstruction
                self.logger.info("Created new HNSW index")
            return self.hnsw_index
        except Exception as e:
            self.logger.exception(f"Failed to load HNSW index: {e}")
            raise

    def save_l2_index(self):
        try:
            index_path = self.vectorstore_base_path / self.vectordb_config.get('FAISS_L2_INDEX', 'faiss_l2.index')
            faiss.write_index(self.l2_index, str(index_path))
            self.logger.info(f"Saved L2 index to {index_path}")
        except Exception as e:
            self.logger.exception(f"Failed to save L2 index: {e}")
            raise

    def save_hnsw_index(self):
        try:
            index_path = self.vectorstore_base_path / self.vectordb_config.get('FAISS_HNSW_INDEX', 'faiss_hnsw.index')
            faiss.write_index(self.hnsw_index, str(index_path))
            self.logger.info(f"Saved HNSW index to {index_path}")
        except Exception as e:
            self.logger.exception(f"Failed to save HNSW index: {e}")
            raise

class VectorDBIngester(VectorDBManager):
    """Class to handle ingestion of embeddings into VectorDB."""

    def __init__(self):
        super().__init__()
        self.load_l2_index()
        self.load_hnsw_index()

    def add_embeddings(self, embeddings):
        """Add embeddings to the vector store."""
        try:
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings).astype('float32')
            else:
                embeddings = embeddings.astype('float32')

            self.l2_index.add(embeddings)
            self.hnsw_index.add(embeddings)
            self.logger.info(f"Added {embeddings.shape[0]} embeddings to vector store")
            self.save_l2_index()
            self.save_hnsw_index()
        except Exception as e:
            self.logger.exception(f"Failed to add embeddings to vector store: {e}")
            raise

class VectorDBRetriever(VectorDBManager):
    """Class to handle retrieval of embeddings from VectorDB."""

    def __init__(self):
        super().__init__()
        self.load_l2_index()
        self.load_hnsw_index()

    def search_embedding(self, embedding_vector, top_n=5, use_l2=True):
        try:
            if use_l2:
                index = self.l2_index
            else:
                index = self.hnsw_index

            embedding_vector = np.array(embedding_vector).reshape(1, -1).astype('float32')
            distances, indices = index.search(embedding_vector, top_n)
            self.logger.info(f"Retrieved {top_n} nearest neighbors")
            return indices.flatten().tolist(), distances.flatten().tolist()
        except Exception as e:
            self.logger.exception(f"Failed to search embedding: {e}")
            raise

class VectorDBDeletor(VectorDBManager):
    """Class to handle deletion of embeddings from VectorDB."""

    def __init__(self):
        super().__init__()
        self.load_l2_index()
        self.load_hnsw_index()

    def delete_embeddings(self, indices):
        self.logger.warning("Deletion is not directly supported in FAISS IndexFlatL2")
        # Implement deletion logic, possibly by rebuilding the index without the specified embeddings
        pass

# Additional model loading functions can be added here if necessary
