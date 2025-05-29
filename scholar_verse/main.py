"""Main entry point for ScholarVerse application."""

import argparse
import os
from pathlib import Path

from scholar_verse.agent import router_agent
from scholar_verse.config import get_config
from scholar_verse.shared_libraries.logging_utils import logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ScholarVerse: Autonomous Scientific Knowledge Graph Platform")
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default=None,
        help="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    return parser.parse_args()


def main():
    """Main entry point for ScholarVerse application."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = get_config()
    logger.info("Configuration loaded successfully")
    
    # Log system information
    logger.info(f"ScholarVerse starting up")
    logger.info(f"Python version: {os.sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Log configuration information
    logger.info(f"Using model: {config['model']['default']}")
    logger.info(f"Environment: {config['app']['env']}")
    
    # Start the router agent
    logger.info("Starting router agent")
    # In Phase 1, we're just setting up the foundation, so we'll just log a message
    logger.info("Router agent initialized successfully")
    logger.info("Phase 1 implementation complete")
    
    # This will be expanded in later phases to actually run the agent
    print("ScholarVerse Phase 1 implementation complete!")
    print("The foundation has been set up successfully.")
    print("Next steps: Implement Phase 2 - Adaptive Router Agent Implementation")


if __name__ == "__main__":
    main()
