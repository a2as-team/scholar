"""Redis cache connector for ScholarVerse."""

from typing import Dict, Any, Optional, Union
import json
import redis

from scholar_verse.config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD
from scholar_verse.shared_libraries.logging_utils import logger


class RedisConnector:
    """Redis cache connector for ScholarVerse.
    
    This class provides a connection to the Redis cache and methods for
    storing and retrieving cached data. It will be expanded in later phases
    to include more sophisticated caching strategies.
    """
    
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, password: Optional[str] = None):
        """Initialize the Redis connector.
        
        Args:
            host: The Redis host, or None to use the value from config.
            port: The Redis port, or None to use the value from config.
            password: The Redis password, or None to use the value from config.
        """
        self.host = host or REDIS_HOST
        self.port = port or REDIS_PORT
        self.password = password or REDIS_PASSWORD
        self.client = None
        logger.info("Initialized Redis connector")
    
    def connect(self) -> bool:
        """Connect to the Redis cache.
        
        Returns:
            True if the connection was successful, False otherwise.
        """
        try:
            logger.info(f"Connecting to Redis cache at {self.host}:{self.port}")
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password if self.password else None,
                decode_responses=True,
            )
            # Verify the connection by pinging the server
            self.client.ping()
            logger.info("Connected to Redis cache successfully")
            return True
        except Exception as e:
            logger.error(f"Error connecting to Redis cache: {e}")
            return False
    
    def close(self) -> None:
        """Close the connection to the Redis cache."""
        if self.client:
            self.client.close()
            self.client = None
            logger.info("Closed Redis cache connection")
    
    def set(self, key: str, value: Union[str, Dict[str, Any]], expiry: Optional[int] = None) -> bool:
        """Set a value in the Redis cache.
        
        Args:
            key: The key to set.
            value: The value to set, which can be a string or a dictionary (will be JSON-encoded).
            expiry: The expiry time in seconds, or None for no expiry.
            
        Returns:
            True if the value was set successfully, False otherwise.
        """
        if not self.client:
            if not self.connect():
                return False
        
        try:
            logger.debug(f"Setting Redis key: {key}")
            
            # Convert dictionary values to JSON strings
            if isinstance(value, dict):
                value = json.dumps(value)
            
            # Set the value
            if expiry:
                self.client.setex(key, expiry, value)
            else:
                self.client.set(key, value)
            
            return True
        except Exception as e:
            logger.error(f"Error setting Redis key: {e}")
            return False
    
    def get(self, key: str, as_json: bool = False) -> Optional[Union[str, Dict[str, Any]]]:
        """Get a value from the Redis cache.
        
        Args:
            key: The key to get.
            as_json: Whether to parse the value as JSON.
            
        Returns:
            The value, or None if the key does not exist or there was an error.
        """
        if not self.client:
            if not self.connect():
                return None
        
        try:
            logger.debug(f"Getting Redis key: {key}")
            
            # Get the value
            value = self.client.get(key)
            
            # Return None if the key does not exist
            if value is None:
                return None
            
            # Parse as JSON if requested
            if as_json:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse Redis value as JSON: {value}")
                    return None
            
            return value
        except Exception as e:
            logger.error(f"Error getting Redis key: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete a key from the Redis cache.
        
        Args:
            key: The key to delete.
            
        Returns:
            True if the key was deleted successfully, False otherwise.
        """
        if not self.client:
            if not self.connect():
                return False
        
        try:
            logger.debug(f"Deleting Redis key: {key}")
            self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting Redis key: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if a key exists in the Redis cache.
        
        Args:
            key: The key to check.
            
        Returns:
            True if the key exists, False otherwise.
        """
        if not self.client:
            if not self.connect():
                return False
        
        try:
            logger.debug(f"Checking if Redis key exists: {key}")
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Error checking if Redis key exists: {e}")
            return False


# Create a default Redis connector
redis_connector = RedisConnector()
