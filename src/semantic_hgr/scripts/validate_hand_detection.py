#!/usr/bin/env python3
"""
Quick validation of hand detection submodule imports.
"""

import sys


def test_imports():
    """Test that all hand detection components can be imported."""
    print("Testing hand detection submodule imports...")

    try:
        # Test data structures
        from semantic_hgr.recognition_engine.hand_detection import HandDetectionResult, HandROI

        print("✓ HandDetectionResult and HandROI imported")

        # Test detector
        from semantic_hgr.recognition_engine.hand_detection import HandDetector

        print("✓ HandDetector imported")

        # Test cropper
        from semantic_hgr.recognition_engine.hand_detection import HandCropper

        print("✓ HandCropper imported")

        # Test initialization
        detector = HandDetector(max_num_hands=1)
        print(f"✓ HandDetector initialized: {detector}")
        detector.close()

        cropper = HandCropper(target_size=(224, 224))
        print(f"✓ HandCropper initialized: {cropper}")

        print("\n✓ All imports successful!")
        return True

    except Exception as e:
        print(f"\n✗ Import failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Entry point for the validate-hand-detection console script."""
    success = test_imports()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
