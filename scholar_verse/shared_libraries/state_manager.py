"""State management for ScholarVerse using Google ADK."""

from typing import Dict, Any, Optional, TypeVar, Generic, Type
from pydantic import BaseModel, Field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

class StateScope(str, Enum):
    """Defines the scope of a state variable."""
    SESSION = "session"  # Specific to current session
    USER = "user"        # Shared across user's sessions
    APP = "app"          # Shared across all users
    TEMP = "temp"        # Temporary, not persisted

class StateKey(str):
    """Represents a state key with scope awareness."""
    
    def __new__(cls, key: str, scope: StateScope = StateScope.SESSION):
        """Create a new state key with the given scope."""
        if scope == StateScope.TEMP:
            prefix = "temp:"
        elif scope == StateScope.USER:
            prefix = "user:"
        elif scope == StateScope.APP:
            prefix = "app:"
        else:  # SESSION
            prefix = ""
        return super().__new__(cls, f"{prefix}{key}")
    
    @property
    def scope(self) -> StateScope:
        """Get the scope of this state key."""
        if self.startswith("temp:"):
            return StateScope.TEMP
        elif self.startswith("user:"):
            return StateScope.USER
        elif self.startswith("app:"):
            return StateScope.APP
        return StateScope.SESSION
    
    @property
    def base_key(self) -> str:
        """Get the base key without the scope prefix."""
        if self.scope != StateScope.SESSION:
            return self.split(":", 1)[1]
        return self

class StateManager:
    """Manages state for ScholarVerse with scoped state support."""
    
    def __init__(self, session_manager: 'SessionManager', user_id: str, session_id: str):
        """Initialize the state manager.
        
        Args:
            session_manager: The session manager instance.
            user_id: The ID of the current user.
            session_id: The ID of the current session.
        """
        self.session_manager = session_manager
        self.user_id = user_id
        self.session_id = session_id
        self._session = None
    
    async def initialize(self):
        """Initialize the state manager by loading the session."""
        if self._session is None:
            self._session = await self.session_manager.get_session(
                user_id=self.user_id,
                session_id=self.session_id
            )
            if self._session is None:
                raise ValueError(f"Session {self.session_id} not found for user {self.user_id}")
    
    def _get_scope_prefix(self, scope: StateScope) -> str:
        """Get the prefix for a given scope."""
        if scope == StateScope.TEMP:
            return "temp:"
        elif scope == StateScope.USER:
            return "user:"
        elif scope == StateScope.APP:
            return "app:"
        return ""
    
    async def get(
        self,
        key: str,
        default: Any = None,
        scope: StateScope = StateScope.SESSION,
        model: Optional[Type[T]] = None
    ) -> Any:
        """Get a state value by key.
        
        Args:
            key: The key to get.
            default: Default value if key doesn't exist.
            scope: The scope of the key.
            model: Optional Pydantic model to parse the value into.
            
        Returns:
            The value associated with the key, or default if not found.
        """
        await self.initialize()
        state_key = StateKey(key, scope)
        
        # Get the value from the session state
        value = self._session.state.get(state_key, default)
        
        # Parse into model if provided
        if model is not None and value is not None:
            if isinstance(value, dict):
                return model.model_validate(value)
            elif isinstance(value, model):
                return value
            else:
                return model(value)
        return value
    
    async def set(
        self,
        key: str,
        value: Any,
        scope: StateScope = StateScope.SESSION,
        merge: bool = False
    ) -> None:
        """Set a state value by key.
        
        Args:
            key: The key to set.
            value: The value to set.
            scope: The scope of the key.
            merge: If True, merge with existing value (for dictionaries).
        """
        await self.initialize()
        state_key = StateKey(key, scope)
        
        # Handle merging dictionaries if requested
        if merge and isinstance(value, dict) and state_key in self._session.state:
            current = self._session.state[state_key]
            if isinstance(current, dict):
                value = {**current, **value}
        
        # Update the state
        updates = {state_key: value}
        self._session = await self.session_manager.update_session_state(
            session=self._session,
            state_updates=updates
        )
    
    async def update(
        self,
        updates: Dict[str, Any],
        scope: StateScope = StateScope.SESSION
    ) -> None:
        """Update multiple state values at once.
        
        Args:
            updates: Dictionary of key-value pairs to update.
            scope: The scope of the keys.
        """
        await self.initialize()
        scoped_updates = {
            StateKey(k, scope): v
            for k, v in updates.items()
        }
        self._session = await self.session_manager.update_session_state(
            session=self._session,
            state_updates=scoped_updates
        )
    
    async def delete(self, key: str, scope: StateScope = StateScope.SESSION) -> bool:
        """Delete a state value by key.
        
        Args:
            key: The key to delete.
            scope: The scope of the key.
            
        Returns:
            True if the key was deleted, False if it didn't exist.
        """
        await self.initialize()
        state_key = StateKey(key, scope)
        
        if state_key in self._session.state:
            # Create a new state dict without the key
            new_state = {
                k: v for k, v in self._session.state.items()
                if k != state_key
            }
            self._session = await self.session_manager.update_session_state(
                session=self._session,
                state_updates=new_state,
                merge=False
            )
            return True
        return False
    
    async def clear_scope(self, scope: StateScope) -> None:
        """Clear all state for a given scope.
        
        Args:
            scope: The scope to clear.
        """
        await self.initialize()
        prefix = self._get_scope_prefix(scope)
        
        # Create a new state dict without keys in the specified scope
        new_state = {
            k: v for k, v in self._session.state.items()
            if not k.startswith(prefix)
        }
        
        if len(new_state) != len(self._session.state):
            self._session = await self.session_manager.update_session_state(
                session=self._session,
                state_updates=new_state,
                merge=False
            )
    
    def get_session_state(self) -> Dict[str, Any]:
        """Get the raw session state dictionary.
        
        Returns:
            The raw session state dictionary.
        """
        if self._session is None:
            raise RuntimeError("StateManager not initialized. Call initialize() first.")
        return self._session.state
