"""Setup script for ma-planning."""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split("\n")
    requirements = [
        r.strip() for r in requirements if r.strip() and not r.startswith("#")
    ]
else:
    requirements = [
        "requests>=2.28.0",
        "pyperplan>=2.0.0",
    ]

setup(
    name="ma-planning",
    version="0.1.0",
    description="Multi-Agent PDDL Planning with LLMs",
    author="MA-Planning Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "ma-plan=cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
