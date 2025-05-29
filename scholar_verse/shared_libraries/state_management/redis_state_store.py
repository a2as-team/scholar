"""
Redis-based state store implementation with versioning support.
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Type, TypeVar, cast

import redis
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)

# Environment variables
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Type variable for state models
StateT = TypeVar('StateT', bound='VersionedState')


class StateVersionMismatchError(Exception):
    """Raised when there's a version mismatch during state updates."""
    pass


class VersionedState(BaseModel):
    """Base class for versioned state objects."""
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict[str, Any] = {}

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    @classmethod
    def from_dict(cls: Type[StateT], data: Dict[str, Any]) -> StateT:
        """Create a state object from a dictionary."""
        if 'data' not in data:
            data = {'data': data}
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the state to a dictionary."""
        return self.dict()


class RedisStateStore:
    """Redis-based state store with versioning support."""
    
    def __init__(self, prefix: str = "scholarverse:state:"):
        """Initialize the Redis state store.
        
        Args:
            prefix: Prefix for all Redis keys
        """
        self.prefix = prefix
        self.redis = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD or None,
            db=REDIS_DB,
            decode_responses=True
        )
        self._check_connection()
    
    def _check_connection(self) -> None:
        """Check Redis connection and log the status."""
        try:
            self.redis.ping()
            logger.info("Successfully connected to Redis")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _get_key(self, key: str) -> str:
        """Get the full Redis key with prefix."""
        return f"{self.prefix}{key}"
    
    async def get_state(self, state_id: str, state_type: Type[StateT]) -> Optional[StateT]:
        """Get a state object by ID.
        
        Args:
            state_id: Unique identifier for the state
            state_type: The state class (must be a subclass of VersionedState)
            
        Returns:
            The state object if found, None otherwise
        """
        try:
            key = self._get_key(state_id)
            data = self.redis.get(key)
            if not data:
                return None
                
            state_data = json.loads(data)
            return state_type.from_dict(state_data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode state data for {state_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting state {state_id}: {e}", exc_info=True)
            return None
    
    async def save_state(self, state_id: str, state: StateT) -> bool:
        """Save a state object.
        
        Args:
            state_id: Unique identifier for the state
            state: The state object to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            key = self._get_key(state_id)
            state.updated_at = datetime.now(timezone.utc)
            
            # Convert to JSON-serializable dict
            state_dict = state.dict()
            
            # Store in Redis with TTL (1 week by default)
            return self.redis.set(
                key,
                json.dumps(state_dict, default=str),
                ex=3600 * 24 * 7  # 1 week TTL
            )
            
        except Exception as e:
            logger.error(f"Error saving state {state_id}: {e}", exc_info=True)
            return False
    
    async def update_state(
        self,
        state_id: str,
        state_type: Type[StateT],
        update_fn: callable,
        *args,
        **kwargs
    ) -> Optional[StateT]:
        """Update a state object atomically.
        
        Args:
            state_id: Unique identifier for the state
            state_type: The state class
            update_fn: Function that takes the current state and returns the updated state
            *args, **kwargs: Additional arguments to pass to update_fn
            
        Returns:
            The updated state if successful, None otherwise
        """
        with self.redis.pipeline() as pipe:
            while True:
                try:
                    # Watch the key for changes
                    pipe.watch(self._get_key(state_id))
                    
                    # Get current state
                    current_data = pipe.get(self._get_key(state_id))
                    current_state = (
                        state_type.from_dict(json.loads(current_data))
                        if current_data
                        else state_type()
                    )
                    
                    # Apply updates
                    updated_state = update_fn(current_state, *args, **kwargs)
                    
                    # Version check
                    if current_state and updated_state.version != current_state.version:
                        raise StateVersionMismatchError(
                            f"Version mismatch: {updated_state.version} != {current_state.version}"
                        )
                    
                    # Save the updated state
                    updated_state.version = self._get_next_version(updated_state.version)
                    updated_state.updated_at = datetime.now(timezone.utc)
                    
                    # Start a transaction
                    pipe.multi()
                    pipe.set(
                        self._get_key(state_id),
                        json.dumps(updated_state.dict(), default=str),
                        ex=3600 * 24 * 7  # 1 week TTL
                    )
                    
                    # Execute the transaction
                    pipe.execute()
                    return updated_state
                    
                except redis.WatchError:
                    # Another client changed the key, retry
                    continue
                except Exception as e:
                    logger.error(f"Error updating state {state_id}: {e}", exc_info=True)
                    return None
    
    def _get_next_version(self, current_version: str) -> str:
        """Increment the version number.
        
        Args:
            current_version: Current version string (e.g., "1.0.0")
            
        Returns:
            Next version string (e.g., "1.0.1")
        """
        try:
            major, minor, patch = map(int, current_version.split('.'))
            return f"{major}.{minor}.{patch + 1}"
        except (ValueError, AttributeError):
            # If version format is invalid, start with 1.0.0
            return "1.0.0"
    
    async def delete_state(self, state_id: str) -> bool:
        """Delete a state object.
        
        Args:
            state_id: Unique identifier for the state
            
        Returns:
            True if successful, False otherwise
        """
        try:
            return bool(self.redis.delete(self._get_key(state_id)))
        except Exception as e:
            logger.error(f"Error deleting state {state_id}: {e}", exc_info=True)
            return False
    
    async def list_states(self, pattern: str = "*") -> list[str]:
        """List all state IDs matching a pattern.
        
        Args:
            pattern: Redis pattern to match (default: "*")
            
        Returns:
            List of matching state IDs (without prefix)
        """
        try:
            keys = self.redis.keys(self._get_key(pattern))
            prefix_len = len(self.prefix)
            return [key[prefix_len:] for key in keys]
        except Exception as e:
            logger.error(f"Error listing states: {e}", exc_info=True)
            return []


# Global instance
state_store = RedisStateStore()
