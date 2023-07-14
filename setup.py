from setuptools import setup

setup(
    name="clipcraft",
    version="1.1",
    description="A package for CLIP-based image and text processing.",
    author='Morgan McKenzie',
    author_email='morgancmckenziecs@gmail.com'
    packages=["clipcraft"],
    install_requires=[
        "transformers",
        "Pillow",
        "numpy",
        "requests",
    ],
)
