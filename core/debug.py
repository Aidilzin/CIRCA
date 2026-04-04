"""
core/debug.py
-------------
Execution tracing utilities for the CIRCA pipeline.
"""

import functools
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


def trace_execution(func: Callable) -> Callable:
    """
    Decorator that logs the entry and exit of a function with millisecond timestamps.
    Uses functools.wraps to preserve function signatures (required for PyQt6 signals).
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Get class name if it's a method
        cls_name = ""
        if args and hasattr(args[0], "__class__"):
            cls_name = f"{args[0].__class__.__name__}."

        func_name = f"{cls_name}{func.__name__}"

        start_time = time.time()
        timestamp = time.strftime("%H:%M:%S", time.localtime(start_time))
        ms = int((start_time % 1) * 1000)

        logger.debug("[%s.%03d] [ENTER] %s", timestamp, ms, func_name)

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            timestamp = time.strftime("%H:%M:%S", time.localtime(end_time))
            ms = int((end_time % 1) * 1000)
            logger.debug("[%s.%03d] [EXIT]  %s", timestamp, ms, func_name)

    return wrapper
