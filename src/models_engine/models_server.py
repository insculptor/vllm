# src/models_engine/models_server.py
import os
import sys
from typing import AsyncGenerator

from fastapi import FastAPI

from src.models_engine.api.models_manager import ModelsManager
from src.models_engine.api.routes import router
from src.utils.config import ConfigLoader

# Add project root to system path
sys.path.insert(0, os.getenv('ROOT_DIR'))

async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Lifespan event handler to initialize and shut down resources."""
    models_manager = ModelsManager()  # Singleton instance, ensures single load
    yield  # App is running now
    await models_manager.shutdown()  # Clean up models

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Models API",
    description="This API provides embedding, reranking, and summarization models hosted on GPU.",
    version="1.0.0",
    lifespan=lifespan
)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    config = ConfigLoader()
    host = config.get("apiserver")["host"]
    port = config.get("apiserver")["models_port"]
    uvicorn.run(app, host=host, port=port)
