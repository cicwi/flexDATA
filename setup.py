from setuptools import setup, find_packages

setup(
    name="flexdata",
    package_dir={'flexdata': 'flexdata'},
    packages=find_packages(),

    install_requires=[
        "numpy",
        "pyqtgraph",
        "astra-toolbox",
        "matplotlib",
        "tqdm",
        "imageio",
        "psutil",
        "scipy",
        "toml",
        ],

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
            'myst-parser'
        ]
    },
    version='1.0.1',
    description='IO routines for CT data',
    url='https://github.com/cicwi/flexdata'
)
