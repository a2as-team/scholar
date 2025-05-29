"""Configuration module for ScholarVerse."""

import logging
import logging.config
import os
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import load_dotenv

# Configure logging
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO',
            'stream': 'ext://sys.stdout'
        },
    },
    'loggers': {
        'scholar_verse': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'WARNING'
    }
})

# Create logger
logger = logging.getLogger('scholar_verse')

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Google Cloud and API Configuration
GOOGLE_GENAI_USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "1") == "1"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

# Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Vertex AI RAG Engine Configuration
RAG_CORPUS = os.getenv("RAG_CORPUS", "")

# Application Configuration
APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Model Configuration
DEFAULT_MODEL = "gemini-1.5-pro"

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
TEMP_DIR = ROOT_DIR / "temp"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)


def get_config() -> Dict[str, Any]:
    """Return the configuration as a dictionary."""
    return {
        "google_genai_use_vertexai": GOOGLE_GENAI_USE_VERTEXAI,
        "google_cloud_project": GOOGLE_CLOUD_PROJECT,
        "google_cloud_location": GOOGLE_CLOUD_LOCATION,
        "neo4j_uri": NEO4J_URI,
        "neo4j_username": NEO4J_USERNAME,
        "redis_host": REDIS_HOST,
        "redis_port": REDIS_PORT,
        "qdrant_host": QDRANT_HOST,
        "qdrant_port": QDRANT_PORT,
        "openai_api_key": OPENAI_API_KEY is not None,
        "openai_model": OPENAI_MODEL,
        "openai_embedding_model": OPENAI_EMBEDDING_MODEL,
        "openai_api_base": OPENAI_API_BASE,
        "openai_api_type": OPENAI_API_TYPE,
        "openai_api_version": OPENAI_API_VERSION,
        "openai_deployment_name": OPENAI_DEPLOYMENT_NAME,
        "openai_embedding_deployment_name": OPENAI_EMBEDDING_DEPLOYMENT_NAME,
        "openai_embedding_api_base": OPENAI_EMBEDDING_API_BASE,
        "openai_embedding_api_key": OPENAI_EMBEDDING_API_KEY is not None,
        "openai_embedding_api_version": OPENAI_EMBEDDING_API_VERSION,
        "openai_embedding_api_type": OPENAI_EMBEDDING_API_TYPE,
        "openai_embedding_dimensions": OPENAI_EMBEDDING_DIMENSIONS,
        "openai_embedding_chunk_size": OPENAI_EMBEDDING_CHUNK_SIZE,
        "openai_embedding_chunk_overlap": OPENAI_EMBEDDING_CHUNK_OVERLAP,
    }


def set_log_level(level: str = "INFO") -> None:
    """Set the log level for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
