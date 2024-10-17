# vllm_openai_server.py

import os
import subprocess
import sys

# Add root directory to Python path
sys.path.insert(0, os.getenv('ROOT_DIR'))

# Import config and logger utilities
from src.utils.config import ConfigLoader
from src.utils.logger import setup_logger

# Load configuration and set up logger
config = ConfigLoader()
log_file_name = config.get("logger")["vllm_log_file"]
logger = setup_logger(log_file_name, logger_name=__name__)

# Retrieve paths from config
root_dir = config["dirs"]["ROOT_DIR"]
cache_dir = config["dirs"]["CACHE_DIR"]

# Retrieve model and engine parameters from config
engine_args = config.get("engine_args", {})
model_path = engine_args["model"]
load_format = engine_args["load_format"]
tensor_parallel_size = engine_args["tensor_parallel_size"]
max_num_batched_tokens = engine_args["max_num_batched_tokens"]
dtype = engine_args["dtype"]
gpu_memory_utilization = engine_args["gpu_memory_utilization"]
max_model_len = engine_args["max_model_len"]
swap_space = engine_args["swap_space"]
trust_remote_code = engine_args.get("trust_remote_code", True)
tokenizer_mode = engine_args.get("tokenizer_mode", "auto")
device = engine_args.get("device", "auto")

# Retrieve server parameters from config
host = config["apiserver"]["host"]
port = config["apiserver"]["vllm_port"]
uvicorn_log_level = config["logger"]["level"]

# Construct the vllm serve command
command = [
    "vllm", "serve",
    model_path,
    "--tokenizer", model_path,
    "--host", host,
    "--port", str(port),
    "--uvicorn-log-level", uvicorn_log_level,
    "--gpu-memory-utilization", str(gpu_memory_utilization),
    "--max-model-len", str(max_model_len),
    "--device", device,
    "--load-format", load_format,
    "--tensor-parallel-size", str(tensor_parallel_size),
    "--max-num-batched-tokens", str(max_num_batched_tokens),
    "--dtype", dtype,
    "--swap-space", str(swap_space),
]

# Optional flags based on config
if trust_remote_code:
    command.append("--trust-remote-code")

if tokenizer_mode:
    command.extend(["--tokenizer-mode", tokenizer_mode])

# Print the command for debugging
logger.info(f"Running command: {' '.join(command)}")

# Run the vllm serve command
try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    logger.error(f"Error: {e}")
    raise RuntimeError("Failed to start the vLLM server.") from e
