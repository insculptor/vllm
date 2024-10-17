# src/utils/constants.py

from src.utils.config import ConfigLoader

# Load configuration once to avoid repetitive I/O
config = ConfigLoader()

# Constants related to API server and ports
API_HOST = config.get("apiserver", {}).get("host", "localhost")
VLLM_PORT = config.get("apiserver", {}).get("vllm_port", 8000)
MODELS_PORT = config.get("apiserver", {}).get("models_port", 8001)

# Construct API URLs
VLLM_API_URL = f"http://{API_HOST}:{VLLM_PORT}"
MODELS_API_URL = f"http://{API_HOST}:{MODELS_PORT}"
VLLM_CHAT_API_URL = f"http://{API_HOST}:{VLLM_PORT}/v1/chat/completions"
EMBEDDING_API_URL = f"http://{API_HOST}:{MODELS_PORT}/v1/embeddings"
RERANKER_API_URL = f"http://{API_HOST}:{MODELS_PORT}/v1/reranker"
SUMMARIZER_API_URL = f"http://{API_HOST}:{MODELS_PORT}/v1/summarize"


# Constants for models and vector database
TOP_K = config.get("vectordb", {}).get("TOP_K", 5)
EMBEDDING_MODEL_NAME = config.get("models", {}).get("EMBEDDING_MODEL", "")
RERANKER_MODEL_NAME = config.get("models", {}).get("RERANKER_MODEL", "")
SUMMARIZATION_MODEL_NAME = config.get("models", {}).get("SUMMARIZATION_MODEL", "")

# Summarization parameters
MAX_LENGTH = config.get("summarize", {}).get("MAX_LENGTH", 150)
MIN_LENGTH = config.get("summarize", {}).get("MIN_LENGTH", 30)
LENGTH_PENALTY = config.get("summarize", {}).get("LENGTH_PENALTY", 2.0)
NUM_BEAMS = config.get("summarize", {}).get("NUM_BEAMS", 4)
EARLY_STOPPING = config.get("summarize", {}).get("EARLY_STOPPING", True)

# Logger settings
LOG_DIR = config.get("dirs", {}).get("LOG_DIR", "./logs/")
MODELS_LOG_FILE = config.get("logger", {}).get("models_log_file", "models_engine.log")
VLLM_LOG_FILE = config.get("logger", {}).get("vllm_log_file", "vllm_engine.log")
