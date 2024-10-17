import uvicorn
from fastapi import FastAPI

from src.utils.config import ConfigLoader
from src.utils.logger import setup_logger
from src.vllm_engine.api.routes import router
from src.vllm_engine.api.vllm_engine_manager import lifespan

# Initialize logger and config
logger = setup_logger()
config = ConfigLoader()

# Initialize FastAPI app with lifespan event handler
app = FastAPI(lifespan=lifespan)

# Include routes from routes.py
app.include_router(router)

if __name__ == "__main__":  
    logger.info("Starting Uvicorn server.")
    uvicorn.run(
        app,
        host=config.get('server', {}).get('host', '0.0.0.0'),
        port=config.get('server', {}).get('port', 8000),
        log_level=config.get('logger',{}).get('level', 'INFO'),
        timeout_keep_alive=config.get('server', {}).get('timeout_keep_alive', 5)
    )