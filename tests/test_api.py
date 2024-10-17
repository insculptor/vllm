import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.getenv('ROOT_DIR'))
from src.models_engine.api.models_manager import ModelsManager
from src.models_engine.models_server import app
from src.utils.config import ConfigLoader
from src.utils.logger import setup_logger
from src.vllm_engine.vllm_server import (
    build_vllm_command,
    load_configuration,
    run_vllm_command,
)

client = TestClient(app)


def test_config_loader():
    config = ConfigLoader()
    assert config.get("apiserver")["host"] == "0.0.0.0"

def test_logger_setup():
    logger = setup_logger("test.log", logger_name="test_logger")
    assert logger.name == "test_logger"

## Vllm Server Tests
@patch("src.vllm_engine.vllm_server.ConfigLoader")
@patch("src.vllm_engine.vllm_server.setup_logger")
def test_load_configuration(mock_logger, mock_config_loader):
    """Test if configuration and logger are loaded correctly."""
    config_mock = MagicMock()
    mock_config_loader.return_value = config_mock
    logger_mock = MagicMock()
    mock_logger.return_value = logger_mock

    config, logger = load_configuration()

    assert config == config_mock
    assert logger == logger_mock

def test_build_vllm_command():
    """Test building the vllm command."""
    config = {
        "apiserver": {"host": "0.0.0.0", "vllm_port": 8000, "env": "dev"},
        "logger": {"level": "debug"},
        "engine_args": {
            "model": "/path/to/model",
            "load_format": "safetensors",
            "tensor_parallel_size": 1,
            "max_num_batched_tokens": 4096,
            "dtype": "float16",
            "gpu_memory_utilization": 0.9,
            "max_model_len": 1024,
            "swap_space": 2,
            "trust_remote_code": True,
            "tokenizer_mode": "auto",
        },
    }

    command = build_vllm_command(config)

    expected_command = [
        "vllm", "serve", "/path/to/model",
        "--tokenizer", "/path/to/model",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--uvicorn-log-level", "debug",
        "--gpu-memory-utilization", "0.9",
        "--max-model-len", "1024",
        "--device", "auto",
        "--load-format", "safetensors",
        "--tensor-parallel-size", "1",
        "--max-num-batched-tokens", "4096",
        "--dtype", "float16",
        "--swap-space", "2",
        "--trust-remote-code",
        "--tokenizer-mode", "auto",
    ]

    assert command == expected_command

@patch("subprocess.run")
def test_run_vllm_command(mock_subprocess):
    """Test running the vllm command."""
    logger = MagicMock()
    command = ["vllm", "serve"]

    run_vllm_command(command, logger)

    mock_subprocess.assert_called_once_with(command, check=True)
    logger.info.assert_called_with("Running command: vllm serve")

    # Test exception handling
    mock_subprocess.side_effect = subprocess.CalledProcessError(1, command)
    with pytest.raises(RuntimeError, match="Failed to start the vLLM server."):
        run_vllm_command(command, logger)


## Model Server Tests
def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_embeddings():
    """Test the embeddings generation endpoint."""
    response = client.post("/v1/embeddings", json={"input": ["This is a sample input"]})
    assert response.status_code == 200
    assert "embeddings" in response.json()
    assert isinstance(response.json()["embeddings"], list)

def test_reranker():
    """Test the document reranking endpoint."""
    models_manager = ModelsManager()
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = [1.0, 0.8]  # Example scores
    models_manager.get_reranker_model = MagicMock(return_value=mock_reranker)

    response = client.post("/v1/reranker", json={
        "query": "What is credit risk?",
        "documents": ["Credit risk refers to...", "Loan defaults are part of credit risk."]
    })

    assert response.status_code == 200
    reranked_docs = response.json()["reranked_documents"]
    assert len(reranked_docs) == 2
    assert isinstance(reranked_docs, list)

def test_summarization():
    """Test the summarization endpoint."""
    models_manager = ModelsManager()
    
    # Mock the summarization model
    mock_summarizer = MagicMock()
    mock_tokenizer = MagicMock()

    # Mock generate() to return token IDs and decode() to return a string summary
    mock_summarizer.generate.return_value = [[101, 102, 103]]  # Example token IDs
    mock_tokenizer.decode.return_value = "This is a summarized text."

    # Ensure get_summarization_model() returns the mock objects
    models_manager.get_summarization_model = MagicMock(
        return_value=(mock_tokenizer, mock_summarizer)
    )

    response = client.post("/v1/summarize", json={
        "input_text": "Credit risk is the probability of a financial loss."
    })

    assert response.status_code == 200
    summary = response.json()["summary"]
    assert isinstance(summary, str)
    assert summary == "This is a summarized text."
