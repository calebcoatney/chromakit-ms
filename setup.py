from setuptools import setup, find_packages

setup(
    name="chromakit-ms",
    version="0.1.0",
    description="A python application for GC-MS data analysis and visualization",
    author="Caleb Coatney",
    author_email="caleb.coatney@nrel.gov",
    url="https://github.com/calebcoatney/chromakit-ms",
    license="Apache-2.0",
    packages=find_packages(),
    py_modules=["main"],
    include_package_data=True,
    package_data={
        "": ["*.qss", "*.png", "*.ico"],
    },
    entry_points={
        "console_scripts": [
            "chromakit-ms=main:main",
        ],
    },
    install_requires=[
        "PySide6>=6.0.0",
        "numpy",
        "matplotlib",
        "scipy",
        "pybaselines",
        "rainbow-api",
    ],
    extras_require={
        "ms": ["ms-toolkit-nrel"],
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
