# src/utils/logger.py

import logging
import logging.config

import colorlog

from src.utils.config import ConfigLoader

print(colorlog.__file__)


def setup_logger(log_level: str = 'INFO'):
    """
    Sets up logging with color-coded console output and file logging.

    Args:
        log_level (str): The logging level (e.g., 'DEBUG', 'INFO', 'WARNING', etc.).

    Returns:
        logger: Configured logger for 'vllm-server'.
    """
    # Load configuration
    config_loader = ConfigLoader()
    logger_config = config_loader.get('logger', {})
    log_level = logger_config["level"].upper()
    log_file = logger_config["file"]

    # Define logging configuration
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,  # Keep existing loggers active
        'formatters': {
            'standard': {  # Formatter for file handler
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'color': {  # Formatter for console handler with colorlog
                '()': 'colorlog.ColoredFormatter',
                'format': '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'log_colors': {
                    'DEBUG': 'white',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'bold_red',
                },
                'secondary_log_colors': {},
                'style': '%',
            },
        },
        'handlers': {
            'console': {  # Console handler with color coding
                'class': 'logging.StreamHandler',
                'formatter': 'color',
                'level': log_level,
            },
            'file': {  # File handler
                'class': 'logging.FileHandler',
                'formatter': 'standard',
                'filename': log_file,
                'level': log_level,
            },
        },
        'loggers': {
            'vllm': {  # Logger for vllm
                'handlers': ['console', 'file'],
                'level': log_level,
                'propagate': False,
            },
            'vllm-server': {  # Logger for vllm-server
                'handlers': ['console', 'file'],
                'level': log_level,
                'propagate': False,
            },
            # Include any other loggers you want to configure
        },
        'root': {  # Root logger
            'handlers': ['console', 'file'],
            'level': log_level,
        },
    }

    # Apply logging configuration
    logging.config.dictConfig(logging_config)

    # Get the vllm-server logger
    logger = logging.getLogger('vllm-server')

    # Optionally set the logging level for vllm if needed
    vllm_logger = logging.getLogger('vllm')
    vllm_logger.setLevel(log_level)

    return logger
