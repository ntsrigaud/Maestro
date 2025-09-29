from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="python-input-manager",
    version="1.0.0",
    author="CSU Capstone Project Team",
    author_email="team@csu-capstone.edu",
    description="High-performance Python Input Manager for video capture and preprocessing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/blackbird410/CSU-Capstone-Project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Video :: Capture",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "gpu": [
            "opencv-contrib-python>=4.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "python-input-manager-demo=python_input_manager:main",
        ],
    },
    package_data={
        "": ["*.md", "*.txt"],
    },
    include_package_data=True,
)