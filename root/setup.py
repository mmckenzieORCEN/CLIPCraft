from setuptools import setup

setup(
    name="clipcraft",
    version="1.0",
    description="A package for CLIP-based image and text processing.",
    packages=["clipcraft"],
    download_url='https://github.com/mmckenzieORCEN/CLIPCraft/releases/download/Release/clipcraft-1.0.tar.gz'
    install_requires=[
        "transformers",
        "Pillow",
        "numpy",
        "requests",
    ],
)
