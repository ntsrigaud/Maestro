"""
Shared utilities for the Hand Gesture Recognition System.

This package contains common utilities that are used across all modules:
- logger: Centralized logging system
"""

from .logger import (
    get_logger,
    configure_logging,
    set_log_level,
    debug,
    info,
    warning,
    error,
    critical,
)

__all__ = [
    "get_logger",
    "configure_logging",
    "set_log_level",
    "debug",
    "info",
    "warning",
    "error",
    "critical",
]
