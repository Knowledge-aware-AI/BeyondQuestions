import random
import time
import functools
from loguru import logger

def network_retry(
    max_retries: int = 6,
    initial_delay: float = 1.0,
    max_delay: float = 120.0,
    exponential_base: float = 2.0,
    retry_on_exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying network operations with exponential backoff and jitter.
    
    Args:
        max_retries (int): Maximum number of retry attempts (default: 6)
        initial_delay (float): Initial delay in seconds (default: 1.0)
        max_delay (float): Maximum delay in seconds (default: 120.0)
        exponential_base (float): Base for exponential backoff (default: 2.0)
        retry_on_exceptions: Tuple of exceptions to catch and retry (default: all)
    
    Returns:
        Decorated function with retry logic
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(
                            f"Failed after {max_retries} retries for {func.__name__}: {e}"
                        )
                        raise
                    
                    logger.warning(
                        f"Network error in {func.__name__} (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)
                    # Add jitter to avoid synchronized retries
                    delay += random.uniform(0, 0.1 * delay)
            
            raise last_exception
        
        return wrapper
    return decorator
