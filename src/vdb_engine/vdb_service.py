# /src/vdb_engine/vdb_service.py

import numpy as np

from src.utils.config import ConfigLoader
from src.utils.logger import setup_logger
from src.utils.mongodb_manager import MongoDBManager
from src.vdb_engine.engine.embedding import EmbeddingEngine
from src.vdb_engine.engine.retrieval import (
    VectorDBDeletor,
    VectorDBIngester,
    VectorDBRetriever,
)


class VDBService:
    """Class to manage embedding and vector DB ingestion."""

    def __init__(self):
        # Initialize logger and config
        self.logger = setup_logger(filename=__name__)
        self.config = ConfigLoader()
        self.logger.info("Initializing VDB Service")

        try:
            # Initialize EmbeddingEngine, VectorDBRetriever, VectorDBIngester, MongoDBManager
            self.embedding_engine = EmbeddingEngine()
            self.retriever = VectorDBRetriever()
            self.ingester = VectorDBIngester()
            self.deletor = VectorDBDeletor()
            self.mongo_manager = MongoDBManager()

            # Load Reranker Model
            self.reranker = self.load_reranker_model()
            self.logger.info("VDB Service initialized successfully")
        except Exception as e:
            self.logger.exception(f"Failed to initialize VDB Service: {e}")
            raise

    def load_reranker_model(self):
        """Load the reranker model."""
        try:
            from sentence_transformers import CrossEncoder
            models_config = self.config.get('models', {})
            reranker_model_name = models_config.get('RERANKER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')

            self.logger.info(f"Loading reranker model: {reranker_model_name}")
            reranker = CrossEncoder(reranker_model_name)
            self.logger.info("Reranker model loaded successfully")
            return reranker
        except Exception as e:
            self.logger.exception(f"Failed to load reranker model: {e}")
            raise

    def get_data(self, query):
        """Fetch matching data from FAISS based on user query."""
        try:
            self.logger.info(f"Fetching data for query: {query}")

            # Generate query embedding
            query_embedding = self.embedding_engine.generate_embedding(query)
            self.logger.debug("Query embedding generated")

            # Retrieve top N documents from VectorDB
            retrieval_config = self.config.get('retrieval', {})
            top_n = retrieval_config.get('TOP_N', 10)
            indices, distances = self.retriever.search_embedding(query_embedding, top_n=top_n)
            self.logger.info(f"Retrieved top {top_n} documents from VectorDB")

            # Fetch documents from MongoDB
            collection_name = self.mongo_manager.collection_name
            documents = self.mongo_manager.find_documents(collection_name, {'_id': {'$in': indices}})
            self.logger.info(f"Fetched {len(documents)} documents from MongoDB")

            if not documents:
                self.logger.info("No documents found matching the query")
                return []

            # Re-rank the documents using the reranker model
            top_k = retrieval_config.get('TOP_K', 5)
            reranked_documents = self.rerank_documents(query, documents, top_k=top_k)
            self.logger.info(f"Re-ranked documents and selected top {top_k}")

            return reranked_documents
        except Exception as e:
            self.logger.exception(f"Failed to get data for query '{query}': {e}")
            raise

    def rerank_documents(self, query, documents, top_k=5):
        """Re-rank documents using the reranker model."""
        try:
            # Prepare input for reranker
            texts = [doc.get('content', '') for doc in documents]
            pairs = list(zip([query]*len(texts), texts))

            # Get relevance scores
            scores = self.reranker.predict(pairs)
            self.logger.debug("Relevance scores obtained from reranker")

            # Sort documents based on scores
            scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
            top_docs = [doc for score, doc in scored_docs[:top_k]]
            return top_docs
        except Exception as e:
            self.logger.exception(f"Failed to rerank documents: {e}")
            raise

    def ingest_data(self, document_name, document_text):
        """Ingest data into MongoDB and VectorDB."""
        try:
            self.logger.info(f"Ingesting data for document: {document_name}")

            # Process document: summarize and generate embedding
            summary, embedding = self.embedding_engine.process_document(document_text)
            self.logger.debug("Document processed: summary and embedding generated")

            # Prepare document for MongoDB
            document = {
                '_id': document_name,  # Use document_name as the unique identifier
                'document_name': document_name,
                'content': document_text,
                'summary': summary,
                'embedding': embedding.tolist()  # Convert numpy array to list for MongoDB
            }

            # Insert document into MongoDB
            collection_name = self.mongo_manager.collection_name
            document_id = self.mongo_manager.insert_document(collection_name, document)
            self.logger.info(f"Document inserted into MongoDB with ID: {document_id}")

            # Add embedding to VectorDB
            self.ingester.add_embeddings(np.array([embedding]))
            self.logger.info("Embedding ingested into VectorDB")
        except Exception as e:
            self.logger.exception(f"Failed to ingest data for document '{document_name}': {e}")
            raise

    def delete_data(self, document_name):
        """Delete data from MongoDB and VectorDB based on document_name."""
        try:
            self.logger.info(f"Deleting data for document: {document_name}")

            # Delete document from MongoDB
            collection_name = self.mongo_manager.collection_name
            delete_result = self.mongo_manager.delete_documents(collection_name, {'_id': document_name})
            if delete_result > 0:
                self.logger.info(f"Deleted document '{document_name}' from MongoDB")
            else:
                self.logger.warning(f"Document '{document_name}' not found in MongoDB")

            # Deleting from VectorDB
            # Assume we have a mapping from document_name to vector index
            # This mapping needs to be maintained during ingestion
            vector_index = self.get_vector_index_for_document(document_name)
            if vector_index is not None:
                self.deletor.delete_embeddings([vector_index])
                self.logger.info(f"Deleted embedding for document '{document_name}' from VectorDB")
            else:
                self.logger.warning(f"Vector index for document '{document_name}' not found. Unable to delete from VectorDB.")
        except Exception as e:
            self.logger.exception(f"Failed to delete data for document '{document_name}': {e}")
            raise

    def get_vector_index_for_document(self, document_name):
        """Retrieve the vector index associated with a document."""
        try:
            # Implement logic to get vector index for a given document_name
            # For example, store a mapping in MongoDB or a separate data structure
            # Here, we'll assume the vector index is stored in the document
            collection_name = self.mongo_manager.collection_name
            document = self.mongo_manager.find_documents(collection_name, {'_id': document_name})
            if document:
                vector_index = document[0].get('vector_index')
                return vector_index
            else:
                self.logger.warning(f"Document '{document_name}' not found in MongoDB")
                return None
        except Exception as e:
            self.logger.exception(f"Failed to get vector index for document '{document_name}': {e}")
            raise
