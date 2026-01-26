"""
Gesture Recognition System - CSU Capstone Project

A hand gesture recognition system for computer control.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="gesture-recognition-system",
    version="0.1.0",
    description="Hand Gesture Recognition System for Computer Control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="CSU Capstone Team",
    python_requires=">=3.8",
    # Automatically find all packages in src/ using proper Python naming conventions
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # Include additional files
    include_package_data=True,
    # Dependencies
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
    ],
    # Optional dependencies for development
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
    },
    # Entry points (if you want to create CLI commands later)
    # entry_points={
    #     "console_scripts": [
    #         "gesture-control=input_manager.main:main",
    #     ],
    # },
)
