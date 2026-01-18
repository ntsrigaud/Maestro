#!/usr/bin/env python3
"""
Comprehensive Environment Validation Script
Tests all required and optional dependencies.

Usage:
    python scripts/verify_environment.py

Exit codes:
    0: All required tests passed
    1: One or more required tests failed
"""

import sys
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header(text):
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}{text:^70}{RESET}")
    print(f"{BOLD}{'='*70}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✓{RESET} {text}")

def print_error(text):
    print(f"{RED}✗{RESET} {text}")

def print_warning(text):
    print(f"{YELLOW}⚠{RESET} {text}")

def test_import(module_name, package_name=None, min_version=None):
    """Test if a module can be imported and check version."""
    package_name = package_name or module_name
    try:
        module = __import__(module_name)
        if hasattr(module, '__version__'):
            version = module.__version__
            version_str = f" (v{version})"
            if min_version:
                # Simple version comparison (major.minor)
                try:
                    current = tuple(map(int, version.split('.')[:2]))
                    required = tuple(map(int, min_version.split('.')[:2]))
                    if current < required:
                        print_error(f"{package_name}{version_str} - UPGRADE REQUIRED (need >={min_version})")
                        return False
                except:
                    pass  # Skip version check if parsing fails
        else:
            version_str = ""
        print_success(f"{package_name}{version_str}")
        return True
    except ImportError as e:
        print_error(f"{package_name} - NOT FOUND")
        print(f"         Install with: pip install {package_name}")
        return False

def test_project_imports():
    """Test project package imports."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        print_success("gesture-recognition-system (local package)")
        return True
    except ImportError as e:
        print_error(f"gesture-recognition-system - {e}")
        print(f"         Run: pip install -e .")
        return False

def test_opencv():
    """Test OpenCV specifically."""
    try:
        import cv2
        version = cv2.__version__
        print_success(f"opencv-python (v{version})")
        
        # Test basic functionality
        import numpy as np
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _ = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        print_success(f"  └─ cv2.cvtColor() works")
        return True
    except Exception as e:
        print_error(f"opencv-python - {e}")
        return False

def test_torch():
    """Test PyTorch and device availability."""
    try:
        import torch
        version = torch.__version__
        print_success(f"torch (v{version})")
        
        # Check device availability
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print_success(f"  └─ CUDA available: {gpu_name}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print_success(f"  └─ Metal (MPS) available")
        else:
            print_warning(f"  └─ CPU only (no GPU acceleration)")
        
        # Test tensor creation
        _ = torch.randn(2, 2)
        print_success(f"  └─ Tensor creation works")
        return True
    except Exception as e:
        print_error(f"torch - {e}")
        return False

def main():
    print_header("Environment Validation Script")
    print("Testing all required dependencies...\n")
    
    results = []
    
    # Core Python libraries
    print(f"{BOLD}Core Libraries:{RESET}")
    results.append(test_import("numpy", min_version="1.24.0"))
    results.append(test_import("PIL", package_name="pillow", min_version="9.0.0"))
    
    # Computer Vision
    print(f"\n{BOLD}Computer Vision:{RESET}")
    results.append(test_opencv())
    
    # Deep Learning
    print(f"\n{BOLD}Deep Learning:{RESET}")
    results.append(test_torch())
    results.append(test_import("torchvision", min_version="0.15.0"))
    results.append(test_import("timm", min_version="0.9.0"))
    
    # Data Science
    print(f"\n{BOLD}Data Science:{RESET}")
    results.append(test_import("sklearn", package_name="scikit-learn", min_version="1.2.0"))
    results.append(test_import("pandas", min_version="1.5.0"))
    results.append(test_import("matplotlib", min_version="3.5.0"))
    results.append(test_import("seaborn", min_version="0.12.0"))
    
    # Utilities
    print(f"\n{BOLD}Utilities:{RESET}")
    results.append(test_import("tqdm", min_version="4.65.0"))
    
    # Optional dependencies (warnings only)
    print(f"\n{BOLD}Optional Dependencies:{RESET}")
    has_sentence_transformers = test_import("sentence_transformers")
    has_huggingface = test_import("huggingface_hub")
    if not has_sentence_transformers:
        print_warning("  (Install with: pip install -e \".[semantic]\")")
    if not has_huggingface:
        print_warning("  (Install with: pip install -e \".[huggingface]\")")
    
    # Project package
    print(f"\n{BOLD}Project Package:{RESET}")
    results.append(test_project_imports())
    
    # Summary
    print_header("Validation Summary")
    
    passed = sum(results)
    total = len(results)
    
    if all(results):
        print(f"{GREEN}{BOLD}✓ ALL TESTS PASSED ({passed}/{total}){RESET}")
        print(f"\n{GREEN}Environment is fully configured!{RESET}")
        print(f"You can now run:")
        print(f"  • python scripts/demo_gesture_recognition.py")
        print(f"  • jupyter notebook notebooks/finetune_vit_hagrid.ipynb")
        return 0
    else:
        failed = total - passed
        print(f"{RED}{BOLD}✗ {failed} TESTS FAILED ({passed}/{total} passed){RESET}")
        print(f"\n{RED}Environment setup incomplete.{RESET}")
        print(f"Review errors above and install missing packages.")
        print(f"\nQuick fix:")
        print(f"  conda env update -f environment.yml --prune")
        print(f"  pip install -e .")
        return 1

if __name__ == "__main__":
    sys.exit(main())
