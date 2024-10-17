# src/models_engine/api/schemas.py

from typing import List

from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    """
    Schema for embedding request.
    """
    input: List[str]  # List of texts to generate embeddings for

class RerankerRequest(BaseModel):
    """
    Schema for reranker request.
    """
    query: str  # The query text
    documents: List[str]  # List of documents to be reranked
