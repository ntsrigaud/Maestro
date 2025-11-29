# Log Rotation Strategy

## Overview

The logging system uses **automatic log rotation** with Python's `RotatingFileHandler` instead of creating timestamped files for each initialization.

## Why Rotation Over Timestamped Files?

### Timestamped Files Approach (OLD - Not Used)

```
logs/
├── system_20251130_093015.log
├── system_20251130_093127.log
├── system_20251130_094302.log
├── system_20251130_095511.log
└── ... (grows indefinitely)
```

**Problems:**

- ❌ One file per app restart (cluttered during development)
- ❌ No automatic cleanup (disk space grows indefinitely)
- ❌ Hard to find relevant logs (which file has the error?)
- ❌ Requires manual cleanup scripts
- ❌ Difficult to monitor (need to watch multiple files)

### Log Rotation Approach (NEW - Current)

```
logs/
├── application.log        # Current logs
├── application.log.1      # Previous rotation (most recent backup)
├── application.log.2      # Older backup
├── application.log.3      # Older backup
├── application.log.4      # Older backup
└── application.log.5      # Oldest backup
```

**Benefits:**

- ✅ Single active log file (easy to monitor: `tail -f logs/application.log`)
- ✅ Automatic cleanup (old logs deleted automatically)
- ✅ Predictable disk usage (max_bytes × backup_count)
- ✅ No manual cleanup needed
- ✅ Standard Python practice
- ✅ Works well for development AND production

## How It Works

### Rotation Trigger

When `application.log` reaches the size limit (default: 10MB):

1. `application.log` → renamed to → `application.log.1`
2. `application.log.1` → renamed to → `application.log.2`
3. `application.log.2` → renamed to → `application.log.3`
4. `application.log.3` → renamed to → `application.log.4`
5. `application.log.4` → renamed to → `application.log.5`
6. `application.log.5` → **deleted** (oldest)
7. New empty `application.log` created for new logs

### Default Configuration

```python
from shared.logger import configure_logging
import logging

configure_logging(
    level=logging.INFO,
    enable_file_logging=True,
    max_bytes=10 * 1024 * 1024,  # 10MB per file
    backup_count=5               # 5 backup files
)
```

**Disk Usage:**

- Maximum: 10MB × 6 files = 60MB total
- Current log: `application.log` (0-10MB)
- Backups: 5 files × 10MB = 50MB

## Use Cases

### Development

```python
# Small files, few backups (easy to read through)
configure_logging(
    level=logging.DEBUG,
    enable_file_logging=True,
    max_bytes=5 * 1024 * 1024,   # 5MB per file
    backup_count=3                # 3 backups
)
# Total: 20MB maximum
```

### Production

```python
# Larger files, more backups (comprehensive history)
configure_logging(
    level=logging.INFO,
    enable_file_logging=True,
    max_bytes=20 * 1024 * 1024,  # 20MB per file
    backup_count=10               # 10 backups
)
# Total: 220MB maximum
```

### High-Traffic Application

```python
# More frequent rotation, keep more history
configure_logging(
    level=logging.WARNING,        # Only important messages
    enable_file_logging=True,
    max_bytes=50 * 1024 * 1024,  # 50MB per file
    backup_count=20               # 20 backups
)
# Total: 1GB maximum
```

## Monitoring Logs

### View Current Logs (Real-time)

```bash
# Follow current log file
tail -f logs/application.log

# With color highlighting (if available)
tail -f logs/application.log | grep --color -E 'ERROR|WARNING|$'
```

### Search All Logs

```bash
# Search current and backup logs
grep "ERROR" logs/application.log*

# Search with context
grep -A 5 -B 5 "MemoryError" logs/application.log*

# Count errors in all logs
grep -c "ERROR" logs/application.log*
```

### Check Disk Usage

```bash
# See size of all log files
ls -lh logs/

# Total size of logs directory
du -sh logs/
```

## Comparison Table

| Feature                   | Timestamped Files     | Log Rotation                    |
| ------------------------- | --------------------- | ------------------------------- |
| Files created per restart | Many (1 per restart)  | One (reuses same file)          |
| Disk space                | Grows indefinitely    | Fixed maximum                   |
| Current log location      | Changes every restart | Always `application.log`        |
| Cleanup needed            | Manual                | Automatic                       |
| Easy to monitor           | No (which file?)      | Yes (`tail -f application.log`) |
| Production ready          | No                    | Yes                             |
| Development friendly      | No (clutter)          | Yes (clean)                     |
| Standard practice         | No                    | Yes (Python stdlib)             |

## Advanced: Time-Based Rotation

If you need rotation based on time instead of size, you can extend the system to use `TimedRotatingFileHandler`:

```python
# Example: Rotate daily at midnight
from logging.handlers import TimedRotatingFileHandler

# In ProjectLogger.initialize():
file_handler = TimedRotatingFileHandler(
    log_file,
    when='midnight',     # Rotate at midnight
    interval=1,          # Every 1 day
    backupCount=30,      # Keep 30 days of logs
    encoding='utf-8'
)
```

**When to use:**

- Long-running servers
- Need logs organized by date
- Compliance requirements (keep 30 days, 90 days, etc.)

**For this project:** Size-based rotation is better for development/testing cycles.

## Log File Locations

### Default (Development)

```
project_root/logs/application.log
```

### Custom (Production)

```python
configure_logging(
    enable_file_logging=True,
    log_dir="/var/log/gesture-recognition"
)
# Creates: /var/log/gesture-recognition/application.log
```

### Per-Environment Config

```python
import os

if os.getenv("ENV") == "production":
    configure_logging(
        level=logging.INFO,
        enable_file_logging=True,
        log_dir="/var/log/gesture-recognition",
        max_bytes=50 * 1024 * 1024,
        backup_count=20
    )
else:
    configure_logging(
        level=logging.DEBUG,
        enable_file_logging=True,  # Or False for dev
        max_bytes=5 * 1024 * 1024,
        backup_count=3
    )
```

## Best Practices

1. **Set appropriate size limits**

   - Development: 5-10MB (quick to open/read)
   - Production: 20-50MB (balance between granularity and file count)

2. **Choose backup count wisely**

   - Development: 3-5 backups (recent history)
   - Production: 10-20 backups (longer history for debugging)

3. **Monitor disk usage**

   - Calculate max usage: `max_bytes × (backup_count + 1)`
   - Ensure enough free space on disk

4. **Use appropriate log levels**

   - DEBUG in development (rotates frequently)
   - INFO/WARNING in production (rotates less often)

5. **Regular monitoring**
   - Set up log monitoring tools
   - Alert on ERROR/CRITICAL messages
   - Archive old logs if needed for compliance

## Summary

**Chosen Strategy:** Size-based log rotation with `RotatingFileHandler`

**Why:**

- Automatic cleanup
- Predictable disk usage
- Easy to monitor
- Standard Python practice
- Works for development and production
- No manual intervention needed

**Configuration:** Flexible via `max_bytes` and `backup_count` parameters
