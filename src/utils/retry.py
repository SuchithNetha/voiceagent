"""
Retry Utility for Sarah Voice Agent.

Provides exponential backoff for API calls to handle transient network
or quota issues gracefully.
"""

import asyncio
import functools
import random
from typing import Type, Union, Tuple, Callable, Any, Optional
from src.utils.logger import setup_logging

logger = setup_logging("Utils-Retry")

def async_retry(
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    tries: int = 3,
    delay: float = 0.5,
    backoff: float = 2.0,
    jitter: bool = True,
    error_callback: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator for retrying async functions with exponential backoff.
    
    Args:
        exceptions: Exception(s) to catch.
        tries: Max number of attempts.
        delay: Initial delay in seconds.
        backoff: Multiplier for delay after each try.
        jitter: If True, add randomness to delay to prevent thundering herd.
        error_callback: Optional function to call on each failure.
    """
    def decorator(f):
        @functools.wraps(f)
        async def wrapper(*args, **kwargs):
            _tries, _delay = tries, delay
            while _tries > 1:
                try:
                    return await f(*args, **kwargs)
                except exceptions as e:
                    wait = _delay * (1 + random.random() if jitter else 1)
                    logger.warning(
                        f"⚠️ {f.__name__} failed: {e}. "
                        f"Retrying in {wait:.2f}s... ({_tries-1} attempts left)"
                    )
                    
                    if error_callback:
                        error_callback(e, _tries)
                        
                    await asyncio.sleep(wait)
                    _tries -= 1
                    _delay *= backoff
                    
            # Last attempt (no try-except here to let the final error bubble up)
            return await f(*args, **kwargs)
        return wrapper
    return decorator

