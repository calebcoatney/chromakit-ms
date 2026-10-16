import re
from pathlib import Path

from setuptools import setup, find_packages

VERSION = re.search(r'__version__ = "([^"]+)"',
                    (Path(__file__).parent / "chromakit_version.py").read_text(encoding="utf-8")).group(1)

setup(
    name="chromakit-ms",
    version=VERSION,
    description="A python application for GC-MS data analysis and visualization",
    author="Caleb Coatney",
    author_email="caleb.coatney@nrel.gov",
    url="https://github.com/calebcoatney/chromakit-ms",
    license="Apache-2.0",
    packages=find_packages(),
    py_modules=["main", "web", "chromakit_version"],
    include_package_data=True,
    package_data={
        "": ["*.qss", "*.png", "*.ico"],
    },
    entry_points={
        "console_scripts": [
            "chromakit-ms=main:main",
            "chromakit-web=web:main",
        ],
    },
    install_requires=[
        "PySide6==6.9.0",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "pybaselines",
        "rainbow-api",
        "uvicorn",
        "fastapi",
        "pydantic",
        "openpyxl",
        "olefile",
    ],
    extras_require={
        # From the GitHub tag: 0.1.3 is not on PyPI yet (account recovery pending). Back to
        # "ms-toolkit-nrel>=0.1.3" once it is.
        "ms": ["ms-toolkit-nrel @ git+https://github.com/calebcoatney/ms-toolkit@v0.1.3"],
        "deconv": ["torch"],
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
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires=">=3.7",
)
