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

def _format_arg(arg: Any) -> str:
    """Format an argument for logging, truncated if necessary, and avoiding bulky data."""
    try:
        # Avoid logging large/bulky objects
        if hasattr(arg, "shape"): # numpy array
            return f"np.ndarray{arg.shape}"
        if hasattr(arg, "width") and hasattr(arg, "height"): # QImage/QPixmap
            return f"Image({arg.width()}x{arg.height()})"
        
        s = repr(arg)
        if len(s) > 100:
            return s[:97] + "..."
        return s
    except Exception:
        return "<unformattable>"

def trace_execution(func: Callable) -> Callable:
    """
    Decorator that logs the entry and exit of a function with millisecond timestamps.
    Enhanced to log arguments and return values (truncated/safely).
    Uses functools.wraps to preserve function signatures (required for PyQt6 signals).
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Get class name if it's a method
        cls_name = ""
        # Check if first arg is 'self'
        if args and hasattr(args[0], "__class__") and func.__name__ in dir(args[0].__class__):
            cls_name = f"{args[0].__class__.__name__}."
            # Strip self from logged args
            logged_args = args[1:]
        else:
            logged_args = args
        
        func_name = f"{cls_name}{func.__name__}"
        
        # Format args/kwargs
        args_str = ", ".join(_format_arg(a) for a in logged_args)
        kwargs_str = ", ".join(f"{k}={_format_arg(v)}" for k, v in kwargs.items())
        params = f"({args_str}{', ' if args_str and kwargs_str else ''}{kwargs_str})"
        
        start_time = time.time()
        timestamp = time.strftime("%H:%M:%S", time.localtime(start_time))
        ms = int((start_time % 1) * 1000)
        
        logger.debug("[%s.%03d] [ENTER] %s %s", timestamp, ms, func_name, params)
        
        try:
            result = func(*args, **kwargs)
            
            # Log successful exit with result
            end_time = time.time()
            timestamp = time.strftime("%H:%M:%S", time.localtime(end_time))
            ms = int((end_time % 1) * 1000)
            formatted_result = _format_arg(result)
            logger.debug("[%s.%03d] [EXIT]  %s -> %s", timestamp, ms, func_name, formatted_result)
            
            return result
        except Exception as e:
            # Log exception before re-raising
            end_time = time.time()
            timestamp = time.strftime("%H:%M:%S", time.localtime(end_time))
            ms = int((end_time % 1) * 1000)
            logger.error("[%s.%03d] [RAISE] %s: %s", timestamp, ms, func_name, e)
            raise
            
    return wrapper