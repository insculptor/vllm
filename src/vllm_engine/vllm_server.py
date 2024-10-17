# src/vllm_engine/vllm_server.py

import os
import subprocess
import sys

sys.path.insert(0, os.getenv('ROOT_DIR'))
from src.utils.config import ConfigLoader
from src.utils.logger import setup_logger


def load_configuration():
    """Loads configuration and initializes logger."""
    config = ConfigLoader()
    logger = setup_logger(config.get("logger")["vllm_log_file"], logger_name=__name__)
    return config, logger

def build_vllm_command(config):
    """Constructs the vllm serve command from configuration."""
    engine_args = config.get("engine_args", {})
    command = [
        "vllm", "serve",
        engine_args["model"],
        "--tokenizer", engine_args["model"],
        "--host", config["apiserver"]["host"],
        "--port", str(config["apiserver"]["vllm_port"]),
        "--uvicorn-log-level", config["logger"]["level"],
        "--gpu-memory-utilization", str(engine_args["gpu_memory_utilization"]),
        "--max-model-len", str(engine_args["max_model_len"]),
        "--device", engine_args.get("device", "auto"),
        "--load-format", engine_args["load_format"],
        "--tensor-parallel-size", str(engine_args["tensor_parallel_size"]),
        "--max-num-batched-tokens", str(engine_args["max_num_batched_tokens"]),
        "--dtype", engine_args["dtype"],
        "--swap-space", str(engine_args["swap_space"]),
    ]

    if engine_args.get("trust_remote_code", True):
        command.append("--trust-remote-code")

    tokenizer_mode = engine_args.get("tokenizer_mode")
    if tokenizer_mode:
        command.extend(["--tokenizer-mode", tokenizer_mode])

    return command

def run_vllm_command(command, logger):
    """Executes the vllm command."""
    logger.info(f"Running command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error: {e}")
        raise RuntimeError("Failed to start the vLLM server.") from e
    
config, logger = load_configuration()
command = build_vllm_command(config)
env = config["apiserver"]["env"]


if __name__ == "__main__":
    if env != "test":
        run_vllm_command(command, logger)
