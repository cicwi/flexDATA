from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(
    name='flexdata',
    version='1.0.1',
    description='I/O routines for CT data',
    url='https://github.com/cicwi/flexdata',
    long_description=readme,
    long_description_content_type='text/markdown',
    license='GNU General Public License v3',
    package_dir={'flexdata': 'flexdata'},
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pyqtgraph',
        'astra-toolbox',
        'matplotlib',
        'tqdm',
        'imageio',
        'psutil',
        'scipy',
        'toml',
        ],
    extras_require={
        'dev': [
            'sphinx',
            'sphinx_rtd_theme',
            'myst-parser'
        ]
    }
)
