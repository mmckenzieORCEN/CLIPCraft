from setuptools import setup


with open('README.md', 'r', encoding='utf-8') as readme_file:
    long_description = readme_file.read()

setup(
    name="clipcraft",
    version="1.4.6",
    description="A package for CLIP-based image and text processing.",
    author='Morgan McKenzie',
    author_email='morgancmckenziecs@gmail.com',
    packages=["clipcraft"],
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mmckenzieORCEN/CLIPCraft',
    install_requires=[
        "transformers",
        "Pillow",
        "numpy",
        "requests",
    ],
)
