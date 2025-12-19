"""
Logging configuration for Agentic Browser.

Provides structured logging with debug mode support.
"""

import logging
import os
import sys
from typing import Optional


def setup_logging(debug: bool = False) -> logging.Logger:
    """Set up logging for agentic browser.
    
    Args:
        debug: Enable debug level logging
        
    Returns:
        Configured logger
    """
    # Check env var first
    if os.getenv("AGENTIC_BROWSER_DEBUG", "").lower() in ("1", "true", "yes"):
        debug = True
    
    level = logging.DEBUG if debug else logging.WARNING
    
    # Configure root logger for agentic_browser
    logger = logging.getLogger("agentic_browser")
    logger.setLevel(level)
    
    # Only add handler if none exist
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        
        # Use different formats for debug vs production
        if debug:
            fmt = "[%(name)s:%(levelname)s] %(message)s"
        else:
            fmt = "%(message)s"
        
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a submodule.
    
    Args:
        name: Logger name (will be prefixed with agentic_browser)
        
    Returns:
        Configured logger
    """
    if not name.startswith("agentic_browser"):
        name = f"agentic_browser.{name}"
    return logging.getLogger(name)
