"""Circuit breaker and retry utilities for resilient agent calls.

This module provides a circuit breaker pattern implementation with retry capabilities
to handle transient failures in agent calls.
"""

import asyncio
import random
import logging
import time
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, Optional, Type, TypeVar, Union, Awaitable, Dict, List, Tuple

# Type variable for the wrapped function
T = TypeVar('T')

class CircuitState(Enum):
    """Possible states of the circuit breaker."""
    CLOSED = auto()    # Normal operation, all calls pass through
    OPEN = auto()      # Circuit is open, all calls fail fast
    HALF_OPEN = auto() # Test if the service has recovered

class CircuitBreakerError(Exception):
    """Exception raised when the circuit is open."""
    def __init__(self, circuit_name: str, state: CircuitState):
        self.circuit_name = circuit_name
        self.state = state
        super().__init__(f"Circuit '{circuit_name}' is {state.name}")

class CircuitBreaker:
    """Circuit breaker pattern implementation with retry capabilities."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: int = 30,
        excluded_exceptions: Tuple[Type[Exception], ...] = ()
    ):
        """Initialize the circuit breaker.
        
        Args:
            name: Name of the circuit breaker for identification
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Time in seconds to wait before attempting recovery
            excluded_exceptions: Exceptions that should not trigger the circuit breaker
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.excluded_exceptions = excluded_exceptions
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._logger = logging.getLogger(f"circuit_breaker.{name}")
    
    @property
    def state(self) -> CircuitState:
        """Get the current state of the circuit."""
        # Check if we should transition from OPEN to HALF_OPEN
        if self._state == CircuitState.OPEN:
            if self._last_failure_time and \
               (datetime.now() - self._last_failure_time).total_seconds() > self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._logger.info(f"Circuit '{self.name}' moved to HALF_OPEN state")
        return self._state
    
    def record_success(self) -> None:
        """Record a successful call and reset the circuit if needed."""
        if self._state == CircuitState.HALF_OPEN:
            self._reset()
            self._logger.info(f"Circuit '{self.name}' reset to CLOSED after successful call")
    
    def record_failure(self, error: Exception) -> None:
        """Record a failed call and update circuit state if needed."""
        # Skip excluded exceptions
        if any(isinstance(error, exc_type) for exc_type in self.excluded_exceptions):
            return
            
        self._failure_count += 1
        self._last_failure_time = datetime.now()
        
        if self._state == CircuitState.HALF_OPEN:
            self._trip()
        elif self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold:
            self._trip()
    
    def _trip(self) -> None:
        """Trip the circuit to open state."""
        self._state = CircuitState.OPEN
        self._logger.warning(f"Circuit '{self.name}' tripped to OPEN state")
    
    def _reset(self) -> None:
        """Reset the circuit to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to wrap a function with circuit breaker logic."""
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            # Check circuit state
            if self.state == CircuitState.OPEN:
                raise CircuitBreakerError(self.name, self.state)
                
            try:
                result = await func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            # Check circuit state
            if self.state == CircuitState.OPEN:
                raise CircuitBreakerError(self.name, self.state)
                
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: float = 0.1,
    retry_on: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that retries a function with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Factor to multiply delay by after each retry
        jitter: Random jitter factor (0.0 to 1.0)
        retry_on: Tuple of exception types to retry on
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    if attempt == max_attempts:
                        break
                        
                    # Calculate next delay with jitter
                    delay = min(delay * backoff_factor, max_delay)
                    jitter_amount = delay * jitter * (2 * random.random() - 1)
                    sleep_time = max(0, delay + jitter_amount)
                    
                    logging.warning(
                        f"Attempt {attempt} failed: {str(e)}. "
                        f"Retrying in {sleep_time:.2f}s..."
                    )
                    
                    await asyncio.sleep(sleep_time)
            
            raise last_exception  # type: ignore
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    if attempt == max_attempts:
                        break
                        
                    # Calculate next delay with jitter
                    delay = min(delay * backoff_factor, max_delay)
                    jitter_amount = delay * jitter * (2 * random.random() - 1)
                    sleep_time = max(0, delay + jitter_amount)
                    
                    logging.warning(
                        f"Attempt {attempt} failed: {str(e)}. "
                        f"Retrying in {sleep_time:.2f}s..."
                    )
                    
                    time.sleep(sleep_time)
            
            raise last_exception  # type: ignore
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator
