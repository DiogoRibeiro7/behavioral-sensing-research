"""Package configuration for sensor_modeling."""

from setuptools import find_packages, setup

setup(
    name="sensor_modeling",
    version="0.1.0",
    author="Diogo Ribeiro",
    author_email="diogo.debastos.ribeiro@gmail.com",
    description="Unified sensor modeling and analysis",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "scikit-learn",
        "networkx",
        "h5py",
        "plotly",
        "bokeh",
        "flask",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "hypothesis",
            "pytest-benchmark",
            "pre-commit",
            "sphinx",
            "sphinx-rtd-theme",
            "nbsphinx",
            "radon",
        ]
    },
    entry_points={
        "console_scripts": [
            "sensor-modeling=sensor_modeling.cli:main",
        ]
    },
    python_requires=">=3.9",
)
