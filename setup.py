"""
Setup script for Spectral Neural Networks (Resonance NN)
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip() 
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="resonance-nn",
    version="2.0.0",
    author="Oluwatosin A. Afolabi",
    author_email="afolabi@genovotech.com",
    description="Spectral Neural Networks: O(n log n) Sequence Modeling with FFT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tafolabi009/RNN",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords=[
        "neural-networks",
        "deep-learning",
        "frequency-domain",
        "spectral-analysis",
        "fft",
        "language-modeling",
        "sequence-modeling",
        "efficient-transformers",
        "attention-alternative",
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
            "pre-commit>=2.20.0",
        ],
        "training": [
            "transformers>=4.30.0",
            "datasets>=2.12.0",
            "wandb>=0.15.0",
            "accelerate>=0.20.0",
            "sentencepiece>=0.1.99",
        ],
        "examples": [
            "matplotlib>=3.5.0",
            "jupyter>=1.0.0",
            "pandas>=1.4.0",
            "seaborn>=0.12.0",
        ],
        "all": [
            # dev
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.20.0",
            # training
            "transformers>=4.30.0",
            "datasets>=2.12.0",
            "wandb>=0.15.0",
            "accelerate>=0.20.0",
            "sentencepiece>=0.1.99",
            # examples
            "matplotlib>=3.5.0",
            "jupyter>=1.0.0",
            "pandas>=1.4.0",
            "seaborn>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "spectral-train=train_production:main",
            "spectral-infer=inference_improved:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
