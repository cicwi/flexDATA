from setuptools import setup, find_packages

setup(
    name="flexdata",
    package_dir={'flexdata': 'flexdata'},
    packages=find_packages(),

    install_requires=[
    "numpy",
    "astra-toolbox",
    "tqdm",
    "imageio",
    "tifffile",
    "psutil",
    "toml",
    "transforms3d",
    "paramiko"],

    version='0.0.1',
)