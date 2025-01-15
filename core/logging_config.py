"""Centralized logging configuration for the Wikipedia Research Assistant."""

import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration for the entire application.
    
    Args:
        log_level: The logging level to use. Defaults to "INFO".
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler(
                'logs/wikipedia_assistant.log',
                maxBytes=10_000_000,  # 10MB
                backupCount=5
            ),
            logging.NullHandler()  # Don't output to terminal by default
        ]
    )
    
    # Create logger for this module
    logger = logging.getLogger(__name__)
    logger.info("Logging setup completed") 