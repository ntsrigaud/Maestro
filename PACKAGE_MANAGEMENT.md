# Quick Reference: Package Management

## One-Time Setup

```bash
# From project root
pip install -e .
```

That's it! No more `sys.path.insert()` needed anywhere.

## Importing Modules

### Before (Fragile)

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.logger import get_logger  # Breaks in different contexts
```

### After (Robust)

```python
from shared.logger import get_logger  # Works everywhere
from input_manager import InputManager
```

## Running Scripts

```bash
# All of these work after pip install -e .
python src/input-manager/example_usage.py
python verify_install.py
python -c "from input_manager import InputManager; print('Works!')"

# Works from any directory
cd docs/
python -c "from shared.logger import get_logger; print('Still works!')"
```

## Verify Installation

```bash
# Check it's installed
pip list | grep gesture-recognition

# Run verification script
python verify_install.py
```

## Common Issues

### "Module not found" error

**Solution:** Run `pip install -e .` from project root

### Changes not reflected

**Solution:** Restart Python interpreter (editable install doesn't require reinstall)

### Wrong Python environment

**Solution:**

```bash
pyenv shell 3.11.9  # or your version
pip install -e .
```

## Adding New Modules

1. Create directory: `src/new-module/`
2. Add `__init__.py`: `src/new-module/__init__.py`
3. Update `setup.py`:

```python
packages=["input_manager", "shared", "new_module"],
package_dir={
    "input_manager": "src/input-manager",
    "shared": "src/shared",
    "new_module": "src/new-module",  # Add this
},
```

4. Reinstall: `pip install -e .`
5. Import: `from new_module import MyClass`

## Why This is Better

| Approach            | sys.path.insert() | PYTHONPATH | pip install -e . |
| ------------------- | ----------------- | ---------- | ---------------- |
| Works in scripts    | Sometimes         | Sometimes  | Always           |
| Works in REPL       | Sometimes         | Sometimes  | Always           |
| Works in tests      | Sometimes         | Sometimes  | Always           |
| Works in IDE        | Sometimes         | Sometimes  | Always           |
| Team-friendly       | No                | No         | Yes              |
| Standard Python     | Hack              | Workaround | Best practice    |
| Virtual env support | Tricky            | Tricky     | Native           |

## Resources

- Full guide: [INSTALL.md](INSTALL.md)
- Python packaging docs: https://packaging.python.org/
- Editable installs: https://pip.pypa.io/en/stable/topics/local-project-installs/
