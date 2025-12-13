#!/usr/bin/env python3
"""
Download semantic gesture recognition model from Google Drive
"""

import sys
from pathlib import Path

try:
    import gdown
except ImportError:
    print("Error: gdown not installed")
    print("Install with: pip install gdown")
    sys.exit(1)


def main():
    # File details
    file_id = "1SBeHhBS30dT6XlzEL4LXlBwxCqa-QeTE"
    output_dir = Path(__file__).parent.parent / "notebooks" / "checkpoints"
    output_file = output_dir / "semantic_model.pth"
    
    print("=" * 70)
    print("SEMANTIC MODEL DOWNLOAD")
    print("=" * 70)
    print(f"Destination: {output_file}")
    print()
    
    # Create directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already exists
    if output_file.exists():
        response = input(f"File already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Download cancelled")
            return
        output_file.unlink()
    
    # Download
    print("Downloading semantic_model.pth from Google Drive...")
    print("This may take a few minutes (~400MB)")
    print()
    
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(output_file), quiet=False)
        
        print()
        print("=" * 70)
        print("✓ Download complete!")
        print(f"✓ Model saved to: {output_file}")
        print()
        print("To run the demo:")
        print("  python scripts/demo_semantic_gesture.py")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nManual download:")
        print("  1. Visit: https://drive.google.com/file/d/1SBeHhBS30dT6XlzEL4LXlBwxCqa-QeTE/view")
        print(f"  2. Download and save to: {output_file}")
        sys.exit(1)


if __name__ == '__main__':
    main()
