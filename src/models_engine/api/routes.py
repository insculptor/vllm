# src/models_engine/api/routes.py

import torch
from fastapi import APIRouter, HTTPException

from src.models_engine.api.models_manager import ModelsManager
from src.models_engine.api.schemas import EmbeddingRequest, RerankerRequest
from src.utils.config import ConfigLoader

router = APIRouter()

# Singleton instance of ModelsManager
models_manager = ModelsManager()
logger = models_manager.logger

config = ConfigLoader()
TOP_K = config.get("vectordb", {}).get("TOP_K", 5) 

@router.get("/health")
def health_check():
    logger.info("Health check requested.")
    return {"status": "ok"}

@router.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest):
    try:
        logger.debug(f"Received input for embedding: {request.input}")
        tokenizer, model = models_manager.get_embedding_model()

        inputs = tokenizer(request.input, padding=True, truncation=True, return_tensors="pt")
        logger.debug(f"Tokenized inputs: {inputs}")

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).tolist()

        logger.info(f"Generated {len(embeddings)} embeddings.")
        return {"embeddings": embeddings}
    except Exception as e:
        logger.error(f"Embedding creation failed: {e}")
        raise HTTPException(status_code=500, detail="Embedding creation failed")

@router.post("/v1/reranker")
async def rerank_documents(request: RerankerRequest):
    try:
        logger.debug(f"Received query: {request.query}")
        logger.debug(f"Received documents: {request.documents}")

        tokenizer, model = models_manager.get_reranker_model()

        query_input = tokenizer(request.query, return_tensors="pt")
        doc_inputs = tokenizer(request.documents, padding=True, truncation=True, return_tensors="pt")

        logger.debug(f"Tokenized query: {query_input}")
        logger.debug(f"Tokenized documents: {doc_inputs}")

        with torch.no_grad():
            query_embedding = model(**query_input).last_hidden_state.mean(dim=1)
            doc_embeddings = model(**doc_inputs).last_hidden_state.mean(dim=1)

        logger.debug(f"Query embedding: {query_embedding}")
        logger.debug(f"Document embeddings: {doc_embeddings}")

        scores = torch.nn.functional.cosine_similarity(query_embedding, doc_embeddings).tolist()
        logger.debug(f"Cosine similarity scores: {scores}")

        sorted_docs = [doc for _, doc in sorted(zip(scores, request.documents), reverse=True)]
        logger.info(f"Reranked {len(sorted_docs)} documents.")

        return {"reranked_documents": sorted_docs[:TOP_K]}
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reranking failed: {str(e)}")
