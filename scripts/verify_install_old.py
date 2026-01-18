#!/usr/bin/env python3
"""
Quick verification script to test that the package is installed correctly.
Run this after: pip install -e .
"""


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from input_manager import InputManager, FrameData

        print("✓ input_manager imports successfully")
    except ImportError as e:
        print(f"✗ Failed to import input_manager: {e}")
        return False

    try:
        from shared.logger import get_logger, configure_logging

        print("✓ shared.logger imports successfully")
    except ImportError as e:
        print(f"✗ Failed to import shared.logger: {e}")
        return False

    try:
        logger = get_logger(__name__)
        logger.info("Logger test message")
        print("✓ Logger works correctly")
    except Exception as e:
        print(f"✗ Logger failed: {e}")
        return False

    return True


def test_input_manager():
    """Test basic InputManager functionality."""
    print("\nTesting InputManager...")

    try:
        from input_manager import InputManager

        manager = InputManager(camera_index=0)
        print(f"✓ InputManager created (Camera {manager._camera_index})")
        return True
    except Exception as e:
        print(f"✗ InputManager test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Package Installation Verification")
    print("=" * 60)

    import_test = test_imports()
    manager_test = test_input_manager()

    print("\n" + "=" * 60)
    if import_test and manager_test:
        print("✓ All tests passed!")
        print("\nYou can now use:")
        print("  from input_manager import InputManager")
        print("  from shared.logger import get_logger")
    else:
        print("✗ Some tests failed")
        print("\nMake sure you ran: pip install -e .")
    print("=" * 60)
