"""Logging utilities for ScholarVerse."""

import logging
import os
from datetime import datetime
from pathlib import Path

from scholar_verse.config import LOG_LEVEL, ROOT_DIR

# Create logs directory if it doesn't exist
LOGS_DIR = ROOT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Configure logging
def setup_logging(name: str = "scholar_verse") -> logging.Logger:
    """Set up logging with the specified name.
    
    Args:
        name: The name of the logger.
        
    Returns:
        A configured logger instance.
    """
    # Create a logger
    logger = logging.getLogger(name)
    
    # Set the logging level based on the environment variable
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Create handlers
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # File handler - create a new log file for each run with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"{name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Create formatters and add them to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Create a default logger
logger = setup_logging()

# Convenience functions
def log_info(message: str) -> None:
    """Log an info message."""
    logger.info(message)

def log_warning(message: str) -> None:
    """Log a warning message."""
    logger.warning(message)

def log_error(message: str) -> None:
    """Log an error message."""
    logger.error(message)

def log_debug(message: str) -> None:
    """Log a debug message."""
    logger.debug(message)

def log_critical(message: str) -> None:
    """Log a critical message."""
    logger.critical(message)
