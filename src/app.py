import uvicorn
from fastapi import FastAPI

from src.api.routes import router
from src.api.vllm_engine_manager import lifespan
from src.utils.config import ConfigLoader
from src.utils.logger import setup_logger

# Initialize logger and config
logger = setup_logger()
config = ConfigLoader()

# Initialize FastAPI app with lifespan event handler
app = FastAPI(lifespan=lifespan)

# Include routes from routes.py
app.include_router(router)

if __name__ == "__main__":
    # Load server configuration    
    logger.info("Starting Uvicorn server.")
    uvicorn.run(
        app,
        host=config.get('server', {}).get('host', '0.0.0.0'),
        port=config.get('server', {}).get('port', 8000),
        log_level=config.get('logger',{}).get('level', 'INFO'),
        timeout_keep_alive=config.get('server', {}).get('timeout_keep_alive', 5)
    )