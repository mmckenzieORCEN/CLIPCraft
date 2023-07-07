from setuptools import setup

setup(
    name="clipcraft",
    version="1.0",
    description="A package for CLIP-based image and text processing.",
    packages=["clipcraft"],
    install_requires=[
        "transformers",
        "Pillow",
        "numpy",
        "requests",
    ],
)
