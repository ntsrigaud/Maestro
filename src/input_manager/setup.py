#!/usr/bin/env python3
"""
DEPRECATED: This setup script is no longer needed.

The project now uses a proper package structure.
Please install the entire project using:

    cd /path/to/CSU-Capstone-Project
    pip install -e .

This will install all modules (input_manager, shared, etc.) properly.

For more information, see:
- INSTALL.md (detailed installation guide)
- PACKAGE_MANAGEMENT.md (quick reference)
- verify_install.py (test installation)
"""

import sys


def main():
    print("=" * 70)
    print("DEPRECATED SETUP SCRIPT")
    print("=" * 70)
    print()
    print("This setup.py is no longer needed.")
    print()
    print("The project now uses proper Python packaging.")
    print("Please install the entire project from the project root:")
    print()
    print("    cd ../../")
    print("    pip install -e .")
    print()
    print("This will properly install all modules:")
    print("  - input_manager")
    print("  - shared (logger, etc.)")
    print("  - future modules")
    print()
    print("After installation, you can import directly:")
    print()
    print("    from input_manager import InputManager")
    print("    from shared.logger import get_logger")
    print()
    print("For more information:")
    print("  - See INSTALL.md for detailed instructions")
    print("  - See PACKAGE_MANAGEMENT.md for quick reference")
    print("  - Run verify_install.py to test installation")
    print()
    print("=" * 70)
    return 1


if __name__ == "__main__":
    sys.exit(main())
