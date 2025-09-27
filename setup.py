"""
Setup script for Elderly Monitoring Model
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="elderly-monitoring-model",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A real-time computer vision system for monitoring elderly individuals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Elderly_Monitoring_Model_Cognitive_Robotics",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "elderly-monitor=src.main_predictor:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["models/*.pkl", "config/*.py"],
    },
)
