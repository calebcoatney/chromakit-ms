from setuptools import setup, find_packages

setup(
    name="chromakit-ms",
    version="0.1.0",
    description="A python application for GC-MS data analysis and visualization",
    author="Caleb Coatney",
    author_email="caleb.coatney@nrel.gov",  # Update with your email
    url="https://github.com/NREL/chromakit-qt",  # Update with actual repository URL
    packages=["ms_toolkit"],  # Import name (with underscore) for Python
    include_package_data=True,
    package_data={
        "": ["*.qss", "*.png", "*.ico"],
    },
    entry_points={
        "console_scripts": [
            "chromakit-qt=main:main",
        ],
    },
    install_requires=[
        "PySide6>=6.0.0",
        "numpy",
        "matplotlib",
        "rainbow-api",
    ],
    extras_require={
        "ms": ["ms-toolkit"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    python_requires=">=3.7",
)
