# Python Naming Convention Migration

## Summary

Migrated from kebab-case to underscore naming for Python package directories, following PEP 8 conventions.

## Changes Made

### Directory Structure

**Before:**

```
src/
├── input-manager/
└── shared/
```

**After:**

```
src/
├── input_manager/
└── shared/
```

### Setup Configuration

**Before (Manual Configuration):**

```python
# setup.py
packages=["input_manager", "shared"],
package_dir={
    "input_manager": "src/input-manager",  # Manual mapping required
    "shared": "src/shared",
}
```

**After (Automatic Discovery):**

```python
# setup.py
packages=find_packages(where="src"),  # Automatic discovery!
package_dir={"": "src"},
```

## Benefits

### 1. Follows Python Standards (PEP 8)

- Package names should be lowercase with underscores
- No special mapping needed between directory and import names
- Consistent with Python ecosystem conventions

### 2. Automatic Package Discovery

- `find_packages()` works out of the box
- No manual package configuration required
- Easier to add new modules (just create directory + `__init__.py`)

### 3. Cleaner Setup Files

- Reduced configuration complexity
- Less maintenance overhead
- Fewer places for errors

### 4. Better IDE Support

- IDEs expect underscore naming
- Better autocomplete and navigation
- Fewer import resolution issues

### 5. Simplified Documentation

- No need to explain directory-to-package mapping
- Import names match directory names
- Less confusion for team members

## Future Module Naming

All new modules should follow this convention:

```
src/recognition_engine/    (not recognition-engine)
src/action_mapper/          (not action-mapper)
src/os_controller/          (not os-controller)
```

## Migration Guide for Team Members

If you have the old structure checked out:

```bash
# Pull latest changes
git pull origin logging-module

# Reinstall the package
pip install -e .

# Verify installation
python verify_install.py
```

Imports remain the same:

```python
from input_manager import InputManager
from shared.logger import get_logger
```

## Technical Details

### Why Hyphens Don't Work

Python identifiers cannot contain hyphens:

```python
# Invalid Python
import input-manager  # SyntaxError!

# Requires mapping
# setup.py must manually map "input-manager" directory to "input_manager" package
```

### Why Underscores Work

Underscores are valid in Python identifiers:

```python
# Valid Python
import input_manager  # Works!

# Directory name matches package name
# find_packages() discovers it automatically
```

### PEP 8 Reference

From [PEP 8 - Package and Module Names](https://www.python.org/dev/peps/pep-0008/#package-and-module-names):

> Modules should have short, all-lowercase names. Underscores can be used
> in the module name if it improves readability. Python packages should
> also have short, all-lowercase names, although the use of underscores
> is discouraged.

We use underscores for multi-word package names (e.g., `input_manager`) as it's more readable than `inputmanager`.

## Verification

Test that everything works:

```bash
# Check package is installed
pip list | grep gesture-recognition

# Test imports
python -c "from input_manager import InputManager; print('✓')"
python -c "from shared.logger import get_logger; print('✓')"

# Run verification script
python verify_install.py
```

## References

- [PEP 8 - Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
- [Python Packaging User Guide](https://packaging.python.org/)
- [setuptools find_packages documentation](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html)
