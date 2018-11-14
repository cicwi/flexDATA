from setuptools import setup, find_packages

setup(
    name="flexdata",
    package_dir={'flexdata': 'flexdata'},
    packages=find_packages(),

    install_requires=[
        "numpy",
        "astra-toolbox",
        "matplotlib",
        "tqdm",
        "imageio",
        "tifffile",
        "psutil",
        "toml",
        "transforms3d",
        "paramiko"],
    extras_require={
        'dev': [
            'autopep8',
            'rope',
            'jedi',
            'flake8',
            'importmagic',
            'autopep8',
            'black',
            'yapf',
            'snakeviz',
            # Documentation
            'sphinx',
            'sphinx_rtd_theme',
            'recommonmark'
        ]
    },
    version='0.0.1',
)
