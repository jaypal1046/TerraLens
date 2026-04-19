from setuptools import setup, find_packages
import os

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="terralens",
    version="0.1.0",
    description="A multi-layered geometric optimizer for non-convex loss landscapes",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="TerraLens Team",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "matplotlib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
