#!/usr/bin/env python3
"""
Quick setup script for InputManager module.
Checks dependencies and provides installation instructions.
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Ensure Python 3.8 or higher."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependency(package_name, import_name=None):
    """Check if a Python package is installed."""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"{package_name} is installed")
        return True
    except ImportError:
        print(f"{package_name} is NOT installed")
        return False


def install_dependencies():
    """Install required dependencies."""
    print("\nInstalling dependencies...")
    requirements_file = Path(__file__).parent / "requirements.txt"

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        )
        print("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("Failed to install dependencies")
        return False


def test_camera_access():
    """Test if camera can be accessed."""
    print("\nTesting camera access...")
    try:
        import cv2

        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print(f"Camera accessible (frame shape: {frame.shape})")
                return True
            else:
                print("Camera opened but failed to read frame")
                return False
        else:
            print("Failed to open camera")
            return False
    except Exception as e:
        print(f"Error accessing camera: {e}")
        return False


def run_quick_test():
    """Run a quick test of InputManager."""
    print("\nRunning quick test...")
    try:
        from input_manager import InputManager

        manager = InputManager(camera_index=0)
        if manager.start_capture():
            frame = manager.get_frame()
            manager.stop_capture()

            if frame is not None:
                print(f"InputManager test passed (frame shape: {frame.shape})")
                return True
            else:
                print("InputManager opened camera but got no frame")
                return False
        else:
            print("InputManager failed to start capture")
            return False
    except Exception as e:
        print(f"InputManager test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run setup and verification."""
    print("=" * 70)
    print("InputManager Setup & Verification")
    print("=" * 70)

    # Check Python version
    print("\n1. Checking Python version...")
    if not check_python_version():
        sys.exit(1)

    # Check dependencies
    print("\n2. Checking dependencies...")
    opencv_installed = check_dependency("opencv-python", "cv2")
    numpy_installed = check_dependency("numpy", "numpy")

    # Install if missing
    if not (opencv_installed and numpy_installed):
        print("\nSome dependencies are missing")
        response = input("Install missing dependencies? (y/n): ")
        if response.lower() == "y":
            if not install_dependencies():
                sys.exit(1)
        else:
            print("Cannot proceed without dependencies")
            print("\nManual installation:")
            print("  pip install -r requirements.txt")
            sys.exit(1)

    # Test camera
    print("\n3. Testing camera access...")
    if not test_camera_access():
        print("\nCamera test failed. Possible issues:")
        print("  - Camera is being used by another application")
        print("  - No camera is connected")
        print("  - Camera permissions not granted (check System Preferences on macOS)")

    # Test InputManager
    print("\n4. Testing InputManager...")
    if run_quick_test():
        print("\n" + "=" * 70)
        print("Setup complete! InputManager is ready to use.")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Run examples: python example_usage.py")
        print("  2. Import in your code: from input_manager import InputManager")
        print("  3. See README.md for API documentation")
    else:
        print("\n" + "=" * 70)
        print("Setup complete but InputManager test failed")
        print("=" * 70)
        print("Please check the errors above and try running example_usage.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
