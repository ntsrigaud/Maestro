# Installation Guide

This guide shows you how to install the Gesture Recognition System as a proper Python package.

## Quick Install (Recommended)

### Option 1: Editable Install (For Development)

This is the **recommended approach** for development. It allows you to make changes to the code without reinstalling:

```bash
# From the project root directory
pip install -e .
```

After this, you can import modules from anywhere:

```python
from input_manager import InputManager
from shared.logger import get_logger
```

### Option 2: Regular Install

For production or testing:

```bash
pip install .
```

### Option 3: Install with Development Tools

To install with additional development dependencies (pytest, black, flake8):

```bash
pip install -e ".[dev]"
```

## What This Does

The editable install (`-e` flag):

- Creates links to your source code instead of copying files
- Changes you make are immediately available without reinstalling
- Properly sets up Python's import system
- Eliminates the need for `sys.path.insert()` hacks
- Works consistently across different execution contexts

## Usage After Installation

### Running Examples

```bash
# From anywhere in the project
python src/input-manager/example_usage.py
```

### Importing in Your Code

```python
# In any Python file in the project
from input_manager import InputManager, FrameData
from shared.logger import get_logger, configure_logging

# Use the modules
manager = InputManager(camera_index=0)
logger = get_logger(__name__)
```

### Running from Different Directories

The editable install means imports work regardless of your current directory:

```bash
# From project root
python -c "from input_manager import InputManager; print('Works!')"

# From any subdirectory
cd docs/
python -c "from input_manager import InputManager; print('Still works!')"

# From src/
cd ../src/
python -c "from input_manager import InputManager; print('Works everywhere!')"
```

## Verifying Installation

Check that the package is installed:

```bash
pip list | grep gesture-recognition
```

You should see:

```
gesture-recognition-system  0.1.0  /path/to/CSU-Capstone-Project/src
```

## Uninstalling

If you need to uninstall:

```bash
pip uninstall gesture-recognition-system
```

## Troubleshooting

### Import Errors After Install

If you still get import errors, ensure you're using the correct Python environment:

```bash
# Check your Python version
python --version

# Check which pip you're using
which pip

# If using pyenv
pyenv shell 3.11.9
pip install -e .
```

### Module Not Found

Make sure you ran the install command from the project root (where `setup.py` and `pyproject.toml` are located):

```bash
cd /Users/neil/dev/repos/CSU-Capstone-Project
pip install -e .
```

### Changes Not Reflected

With editable install (`-e`), code changes should be immediate. If not:

1. Restart your Python interpreter
2. Check you're using the correct Python environment
3. Verify with `pip list | grep gesture-recognition`

## Migration from Old Approach

### Before (Fragile)

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.logger import get_logger  # Might fail depending on execution context
```

### After (Robust)

```python
from shared.logger import get_logger  # Works everywhere after pip install -e .
```

## Alternative: PYTHONPATH (Not Recommended)

If you cannot use pip install for some reason, you can set PYTHONPATH:

```bash
# Add to your ~/.zshrc or ~/.bashrc
export PYTHONPATH="/Users/neil/dev/repos/CSU-Capstone-Project/src:$PYTHONPATH"
```

**However, this approach:**

- Requires manual configuration on every machine
- Doesn't work well with virtual environments
- Can cause conflicts with other projects
- Is harder to maintain in CI/CD pipelines

**The editable install is superior because:**

- Works with virtual environments
- Properly integrates with Python's packaging system
- Easy to share with team members
- Standard Python practice
- Works in all execution contexts (scripts, REPL, IDEs, tests)
