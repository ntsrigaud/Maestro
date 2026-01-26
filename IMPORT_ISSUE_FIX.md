# FrameData Helper Methods - Import Issue Fix

## Problem

The `FrameData` helper methods (`timestamp_ms()`, `time_delta_ms()`, `calculate_fps()`) exist in the source code but are not accessible when importing from the installed package.

## Root Cause

Python is using **cached bytecode** (`.pyc` files) from before the helper methods were added. The package needs to be reinstalled to update the installed version.

## Solution

Reinstall the package to update cached imports:

```bash
cd /Users/neil/dev/repos/CSU-Capstone-Project
pip install -e . --force-reinstall --no-deps
```

Or clear Python cache and reinstall:

```bash
# Clear all __pycache__ directories
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Clear pip cache for this package
pip cache remove gesture-recognition-system

# Reinstall in editable mode
pip install -e .
```

## Verification

After reinstalling, verify the methods are accessible:

```bash
python test_framedata_methods.py
```

Expected output:

```
✓ timestamp_ms           = <value>
✓ timestamp_seconds      = <value>
✓ time_delta_ms         = <value>
✓ calculate_fps         = <value>

✓ All helper methods are accessible!
```

## Alternative: Import Directly from Source

If you can't reinstall right now, the demo script has been updated with compatibility helpers that work by importing directly from source.

## Why This Happened

1. Package was installed with `pip install -e .`
2. Python created bytecode cache in `src/input_manager/__pycache__/`
3. Helper methods were added to `input_manager.py`
4. Python continued using old cached bytecode
5. Import failed with `AttributeError`

## Prevention

After making changes to installed packages:

- Always reinstall: `pip install -e .`
- Or delete `__pycache__` directories
- Or restart Python interpreter
