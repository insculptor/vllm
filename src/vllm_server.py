import json
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from src.utils.config import ConfigLoader
from src.utils.logger import setup_logger

# Initialize config and logger
config = ConfigLoader()
sampling_parameters_config = config.get('sampling_parameters', {})
logger = setup_logger()

async def lifespan(app: FastAPI):
    """Lifespan event handler to initialize and shut down resources."""
    global engine
    # Startup: Initialize the engine
    engine_args_config = config.get('engine_args', {})
    logger.info("Initializing engine with config.")
    logger.debug(f"Engine configuration: {engine_args_config}")
    async_engine_args = AsyncEngineArgs(**engine_args_config)
    engine = AsyncLLMEngine.from_engine_args(async_engine_args)
    logger.info("Engine initialized successfully.")
    yield

    # Shutdown: Stop the engine
    logger.info("Shutting down the application...")
    try:
        logger.debug("Stopping the LLM engine...")
        await engine.shutdown_background_loop()
        logger.debug("LLM engine stopped successfully.")
    except Exception as e:
        logger.error(f"Error occurred during shutdown: {e}")
    finally:
        # Ensure resources are cleaned up even in case of an error
        logger.info("Final cleanup completed.")

# Initialize FastAPI app with a lifespan event
app = FastAPI(lifespan=lifespan)


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



@app.get("/")
def read_root():
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the vLLM server"}


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request."""
    logger.info("Received request for text generation.")
    
    # Log the request data
    request_dict = await request.json()
    logger.debug(f"Request data: {request_dict}")

    # Extract prompt and stream options
    prompt = request_dict.pop("prompt")
    logger.info(f"Prompt received: {prompt}")
    
    stream = request_dict.pop("stream", False)
    logger.info(f"Stream flag: {stream}")
    
    # Extract or use default sampling parameters
    sampling_params = get_sampling_params(request_dict)
    request_id = random_uuid()
    logger.info(f"Generated request ID: {request_id}")

    # Start generating text
    results_generator = engine.generate(prompt, sampling_params, request_id)
    logger.info("Started text generation.")

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        logger.debug(f"Streaming results for request ID: {request_id}")
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            logger.debug(f"Streaming partial result: {ret}")
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        logger.info(f"Streaming enabled for request ID: {request_id}")
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            logger.warning(f"Client disconnected for request ID: {request_id}, aborting.")
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    logger.info(f"Text generation completed for request ID: {request_id}. Final result: {ret}")

    return JSONResponse(ret)


if __name__ == "__main__":
    # Load server configuration
    server_config = config.get('server', {})
    
    logger.info("Starting Uvicorn server.")
    uvicorn.run(
        app,
        host=server_config["host"],
        port=server_config["port"],
        log_level="debug",
        timeout_keep_alive=server_config["timeout_keep_alive"]
    )