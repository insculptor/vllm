# src/utils/logger.py
import logging
import logging.config
import os

import colorlog

from src.utils.config import ConfigLoader

print(colorlog.__file__)

def setup_logger(log_file_name: str = None, logger_name: str = None):
    """
    Sets up logging with color-coded console output and file logging.

    Args:
        log_file_name (str): Optional log file name. If not provided, use the default from YAML config.
        logger_name (str): Name for the logger (e.g., module name). Defaults to 'root'.

    Returns:
        logger: Configured logger.
    """
    # Load configuration
    config_loader = ConfigLoader()
    logger_config = config_loader.get('logger', {})
    log_dir = config_loader.get('dirs')['LOG_DIR']
    log_level = logger_config.get('level', 'INFO').upper()

    # Use provided file name or default from config
    log_file = log_file_name or logger_config.get("file", "vllm_engine.log")
    log_file_path = os.path.join(log_dir, log_file)

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Define logging configuration
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'color': {
                '()': 'colorlog.ColoredFormatter',
                'format': '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'log_colors': {
                    'DEBUG': 'white',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'bold_red',
                },
                'style': '%',
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'color',
                'level': log_level,
            },
            'file': {
                'class': 'logging.FileHandler',
                'formatter': 'standard',
                'filename': log_file_path,
                'level': log_level,
            },
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': log_level,
        },
    }

    # Apply the logging configuration
    logging.config.dictConfig(logging_config)

    # Create and return the logger with the given name
    logger = logging.getLogger(logger_name)
    logger.debug(f"Logger initialized. Log file: {log_file_path}")
    return logger
