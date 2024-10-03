from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

from src.utils.config import ConfigLoader
from src.utils.logger import setup_logger

# Initialize config and logger
config = ConfigLoader()
sampling_parameters_config = config.get('sampling_parameters', {})
logger = setup_logger()

class VLLMEngineManager:
    """Manager class to handle the initialization and retrieval of the vLLM engine."""
    
    _engine = None

    @classmethod
    def initialize_engine(cls):
        """Initializes the engine if it's not already initialized."""
        if cls._engine is None:
            try:
                # Get engine configuration from config
                engine_args_config = config.get('engine_args', {})
                logger.info("Initializing engine with config.")
                logger.debug(f"Engine configuration: {engine_args_config}")
                
                # Initialize the engine
                async_engine_args = AsyncEngineArgs(**engine_args_config)
                cls._engine = AsyncLLMEngine.from_engine_args(async_engine_args)
                logger.info("Engine initialized successfully.")
            except Exception as e:
                logger.error(f"Engine initialization failed: {e}")
                raise RuntimeError("Failed to initialize the engine") from e
    
    @classmethod
    def get_engine(cls):
        """Retrieves the engine instance, initializing it if necessary."""
        if cls._engine is None:
            logger.error("Engine has not been initialized.")
            raise RuntimeError("Engine is not initialized.")
        return cls._engine
            
    @classmethod
    async def shutdown_engine(cls):
        """Shuts down the engine and performs cleanup."""
        if cls._engine is not None:
            try:
                logger.info("Shutting down the engine...")
                logger.debug("LLM engine stopped successfully.")
            except Exception as e:
                logger.error(f"Error occurred during shutdown: {e}")
            finally:
                # Set the engine to None after the shutdown completes
                cls._engine = None
                logger.info("Engine cleanup completed.")
        else:
            logger.info("Engine was not initialized, skipping shutdown.")

async def lifespan(app):
    """Lifespan event handler to initialize and shut down resources."""
    # Startup: Initialize the engine
    VLLMEngineManager.initialize_engine()

    yield  # The server is running now

    # Shutdown: Stop the engine
    await VLLMEngineManager.shutdown_engine()

def get_sampling_params(request_dict: dict) -> SamplingParams:
    """
    If the request contains sampling parameters, use them. 
    Otherwise, load defaults from config.
    """
    sampling_params_data = {}

    for param in ['max_tokens', 'temperature', 'top_p', 'num_beams']:
        if param in request_dict:
            sampling_params_data[param] = request_dict[param]  # Use directly from the request
        elif param in sampling_parameters_config:
            sampling_params_data[param] = sampling_parameters_config[param]  # Fallback to config
    
    logger.debug(f"Sampling parameters used: {sampling_params_data}")
    return SamplingParams(**sampling_params_data)
