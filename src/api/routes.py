import json
from typing import AsyncGenerator

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from vllm.utils import random_uuid

from src.api.vllm_engine_manager import VLLMEngineManager, get_sampling_params
from src.utils.logger import setup_logger

logger = setup_logger()
router = APIRouter()

@router.get("/")
def read_root():
    """Root endpoint"""
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the vLLM server"}

@router.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request."""
    logger.info("Received request for text generation.")

    # Get the global engine instance from VLLMEngineManager
    try:
        engine = VLLMEngineManager.get_engine()
    except RuntimeError as e:
        logger.error(str(e))
        return Response(status_code=500, content="LLM Engine is not initialized.")

    # Log the request data
    request_dict = await request.json()
    logger.debug(f"Request data: {request_dict}")

    # Extract prompt and stream options
    prompt = request_dict.pop("prompt")
    logger.info(f"Prompt received: {prompt}")
    
    stream = request_dict.pop("stream", True)
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
