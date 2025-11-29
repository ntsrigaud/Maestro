"""
Shared logging module for the Hand Gesture Recognition System.

This module provides a centralized logging system that can be used across
all modules (input_manager, recognition_engine, action_mapper, os_controller).

Features:
- Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Colored console output for better readability
- File logging with rotation
- Module-specific loggers
- Consistent formatting across the project
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


# ANSI color codes for console output
class LogColors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"

    # Log level colors
    DEBUG = "\033[36m"  # Cyan
    INFO = "\033[32m"  # Green
    WARNING = "\033[33m"  # Yellow
    ERROR = "\033[31m"  # Red
    CRITICAL = "\033[35m"  # Magenta

    # Component colors
    TIMESTAMP = "\033[90m"  # Gray
    MODULE = "\033[94m"  # Blue


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to console output."""

    LEVEL_COLORS = {
        logging.DEBUG: LogColors.DEBUG,
        logging.INFO: LogColors.INFO,
        logging.WARNING: LogColors.WARNING,
        logging.ERROR: LogColors.ERROR,
        logging.CRITICAL: LogColors.CRITICAL,
    }

    def format(self, record):
        # Save original levelname
        levelname = record.levelname

        # Add colors if outputting to terminal
        if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            level_color = self.LEVEL_COLORS.get(record.levelno, "")
            record.levelname = f"{level_color}{levelname}{LogColors.RESET}"
            record.name = f"{LogColors.MODULE}{record.name}{LogColors.RESET}"

        result = super().format(record)

        # Restore original levelname
        record.levelname = levelname

        return result


class ProjectLogger:
    """
    Centralized logging system for the project.

    Usage:
        from shared.logger import get_logger

        logger = get_logger(__name__)
        logger.info("Starting module")
        logger.warning("Resource constraint detected")
        logger.error("Failed to process frame")
    """

    _initialized = False
    _log_level = logging.INFO
    _log_dir: Optional[Path] = None
    _file_logging_enabled = False

    @classmethod
    def initialize(
        cls,
        level: int = logging.INFO,
        log_dir: Optional[str] = None,
        enable_file_logging: bool = False,
        format_string: Optional[str] = None,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB default
        backup_count: int = 5,  # Keep 5 backup files
        force_reinit: bool = False,  # Allow reinitialization
    ):
        """
        Initialize the project-wide logging system.

        Can be called multiple times. If already initialized, does nothing unless
        force_reinit=True is passed.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files (default: project_root/logs)
            enable_file_logging: Whether to enable file logging
            format_string: Custom format string for log messages
            max_bytes: Maximum size of each log file before rotation (default: 10MB)
            backup_count: Number of backup files to keep (default: 5)
            force_reinit: If True, reinitialize even if already initialized (default: False)
        """
        # Allow reinitialization if explicitly requested
        if cls._initialized and not force_reinit:
            return

        cls._log_level = level
        cls._file_logging_enabled = enable_file_logging

        # Set up log directory
        if log_dir:
            cls._log_dir = Path(log_dir)
        else:
            # Default to project_root/logs
            project_root = Path(__file__).parent.parent.parent
            cls._log_dir = project_root / "logs"

        if enable_file_logging:
            cls._log_dir.mkdir(parents=True, exist_ok=True)

        # Default format
        if format_string is None:
            format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)

        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_formatter = ColoredFormatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # File handler (if enabled)
        if enable_file_logging:
            # Use a single log file with rotation instead of timestamped files
            log_file = cls._log_dir / "application.log"

            # RotatingFileHandler automatically rotates when file reaches max_bytes
            # Keeps backup_count old files (application.log.1, application.log.2, etc.)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                format_string, datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

        cls._initialized = True

    @classmethod
    def reset(cls):
        """
        Reset the logging system to uninitialized state.
        
        Closes all handlers and clears configuration. Useful for testing
        or when you need to completely reconfigure logging.
        """
        root_logger = logging.getLogger()
        
        # Close and remove all handlers
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
        
        # Reset state
        cls._initialized = False
        cls._log_level = logging.INFO
        cls._log_dir = None
        cls._file_logging_enabled = False

    @classmethod
    def set_level(cls, level: int):
        """Change the logging level at runtime."""
        cls._log_level = level
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setLevel(level)

    @classmethod
    def get_level(cls) -> int:
        """Get current logging level."""
        return cls._log_level
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Check if logging system has been initialized."""
        return cls._initialized


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Configured logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Processing started")
    """
    # Initialize with defaults if not already done
    if not ProjectLogger._initialized:
        ProjectLogger.initialize()

    return logging.getLogger(name)


def set_log_level(level: int):
    """
    Set the logging level for all loggers.

    Args:
        level: One of logging.DEBUG, INFO, WARNING, ERROR, CRITICAL

    Example:
        set_log_level(logging.DEBUG)  # Show all messages
        set_log_level(logging.WARNING)  # Only warnings and above
    """
    ProjectLogger.set_level(level)


def configure_logging(
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    enable_file_logging: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    force_reinit: bool = False,
):
    """
    Configure the project logging system.

    Can be called multiple times. Subsequent calls are ignored unless force_reinit=True.

    Args:
        level: Logging level (default: INFO)
        log_dir: Directory for log files
        enable_file_logging: Whether to write logs to files
        max_bytes: Maximum size per log file before rotation (default: 10MB)
        backup_count: Number of rotated backup files to keep (default: 5)
        force_reinit: Force reinitialization even if already configured (default: False)

    Example:
        # Development: verbose console logging only
        configure_logging(level=logging.DEBUG)

        # Production: info level with file logging and rotation
        configure_logging(
            level=logging.INFO,
            enable_file_logging=True,
            log_dir="/var/log/gesture-recognition",
            max_bytes=20 * 1024 * 1024,  # 20MB per file
            backup_count=10  # Keep 10 backup files
        )
        
        # Reconfigure at runtime (e.g., enable file logging dynamically)
        configure_logging(
            level=logging.DEBUG,
            enable_file_logging=True,
            force_reinit=True  # Required to apply new settings
        )
    """
    ProjectLogger.initialize(
        level=level,
        log_dir=log_dir,
        enable_file_logging=enable_file_logging,
        max_bytes=max_bytes,
        backup_count=backup_count,
        force_reinit=force_reinit,
    )


# Convenience functions for quick logging without creating logger instance
_default_logger = None


def _get_default_logger():
    global _default_logger
    if _default_logger is None:
        _default_logger = get_logger("gesture-recognition")
    return _default_logger


def debug(message: str, *args, **kwargs):
    """Quick debug logging."""
    _get_default_logger().debug(message, *args, **kwargs)


def info(message: str, *args, **kwargs):
    """Quick info logging."""
    _get_default_logger().info(message, *args, **kwargs)


def warning(message: str, *args, **kwargs):
    """Quick warning logging."""
    _get_default_logger().warning(message, *args, **kwargs)


def error(message: str, *args, **kwargs):
    """Quick error logging."""
    _get_default_logger().error(message, *args, **kwargs)


def critical(message: str, *args, **kwargs):
    """Quick critical logging."""
    _get_default_logger().critical(message, *args, **kwargs)
