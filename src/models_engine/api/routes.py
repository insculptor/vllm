# src/models_engine/api/routes.py

import torch
from fastapi import APIRouter, HTTPException

import src.utils.constants as c
from src.models_engine.api.models_manager import ModelsManager
from src.models_engine.api.schemas import (
    EmbeddingRequest,
    RerankerRequest,
    SummarizationRequest,
)

router = APIRouter()

# Singleton instance of ModelsManager
models_manager = ModelsManager()
logger = models_manager.logger


@router.get("/health", summary="API Health Check", tags=["Health"])
def health_check():
    """
    Check the health status of the API.

    **Returns:**
    - **status**: A string indicating the health status.

    Example Response:
    ```json
    {
      "status": "ok"
    }
    ```
    """
    logger.info("Health check requested.")
    return {"status": "ok"}


@router.post("/v1/embeddings", summary="Generate Embeddings", tags=["Embeddings"])
async def create_embedding(request: EmbeddingRequest):
    """
    Generate vector embeddings for a list of input texts.

    **Request Body:**
    - **input** (List[str]): List of input strings to generate embeddings for.

    **Response:**
    - **embeddings** (List[List[float]]): List of vector embeddings.

    Example Input:
    ```json
    {
      "input": ["This is a sample text.", "Another text input."]
    }
    ```

    Example Response:
    ```json
    {
      "embeddings": [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
      ]
    }
    ```
    """
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


@router.post("/v1/reranker", summary="Rerank Documents", tags=["Reranker"])
async def rerank_documents(request: RerankerRequest):
    """
    Rerank a set of documents using the CrossEncoder model.

    **Request Body:**
    - **query** (str): Query text to rank documents against.
    - **documents** (List[str]): List of documents to rerank.

    **Response:**
    - **reranked_documents** (List[str]): Top-K documents sorted by relevance.

    Example Input:
    ```json
    {
      "query": "What is credit risk?",
      "documents": [
        "Credit risk refers to...",
        "Loan defaults are part of..."
      ]
    }
    ```

    Example Response:
    ```json
    {
      "reranked_documents": [
        "Credit risk refers to...",
        "Loan defaults are part of..."
      ]
    }
    ```
    """
    try:
        query = request.query
        documents = request.documents
        logger.debug(f"Received query: {query}")
        logger.debug(f"Received documents: {documents}")

        # Get the CrossEncoder model
        reranker_model = models_manager.get_reranker_model()

        # Create input pairs for ranking
        input_pairs = [(query, doc) for doc in documents]
        logger.debug(f"Input pairs: {input_pairs}")

        # Perform ranking
        scores = reranker_model.predict(input_pairs)
        logger.debug(f"Rerank scores: {scores}")

        # Sort documents by score
        sorted_docs = [doc for _, doc in sorted(zip(scores, documents), reverse=True)]
        logger.info(f"Reranked {len(sorted_docs)} documents.")

        return {"reranked_documents": sorted_docs[:c.TOP_K]}
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reranking failed: {str(e)}")


@router.post("/v1/summarize", summary="Summarize Text", tags=["Summarization"])
async def summarize_text(request: SummarizationRequest):
    """
    Summarizes the input text using a summarization model.

    **Request Body:**
    - **input_text** (str): The input text to summarize.

    **Response:**
    - **summary** (str): The summarized version of the input text.

    Example Input:
    ```json
    {
      "input_text": "Credit risk is the probability of a financial loss..."
    }
    ```

    Example Response:
    ```json
    {
      "summary": "Credit risk refers to the probability of financial loss..."
    }
    ```
    """
    try:
        input_text = request.input_text
        logger.debug(f"Received input text for summarization: {input_text}")

        tokenizer, model = models_manager.get_summarization_model()

        # Tokenize the input
        inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
        logger.debug(f"Tokenized input: {inputs}")

        # Generate the summary using parameters from YAML
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=c.MAX_LENGTH,
                min_length=c.MIN_LENGTH,
                length_penalty=c.LENGTH_PENALTY,
                num_beams=c.NUM_BEAMS,
                early_stopping=c.EARLY_STOPPING,
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        logger.info("Generated summary successfully.")
        return {"summary": summary}

    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail="Summarization failed.")
    