# flexDATA

This project is a part of the larger X-ray tomographic reconstruction toolbox comprised of [flexDATA](https://github.com/cicwi/flexDATA), [flexTOMO](https://github.com/cicwi/flexTOMO) and [flexCALC](https://github.com/cicwi/flexCALC).
flexDATA contains IO routines that were originally developed for the Flex-Ray X-ray CT scanner but can be used for reading and writing data from other types of CT scanners. It provides an interface between a scanner and a GPU-based CT reconstruction [ASTRA Toolbox](https://github.com/astra-toolbox/astra-toolbox). It can be especially helpful when the scanner has many degrees of freedom and the geometry of each scan is defined by a large number of parameters.

## Getting Started

It takes a few steps to setup flexDATA on your machine. We recommend that the user installs [conda package manager](https://docs.anaconda.com/miniconda/) for Python 3.

### Installing with conda

`conda install flexdata -c cicwi -c astra-toolbox -c nvidia`

### Installing with pip

`pip install flexdata`

### Installing from source

```bash
git clone https://github.com/cicwi/flexdata.git
cd flexdata
pip install -e .
```

## Running the examples

To learn about the functionality of the package check out our `examples/` folder. Examples are separated into blocks that are best to run in VS Code / Spyder environment step-by-step.

## Modules

flexDATA is comprised of the following modules:

* `flexdata.data`: Read / write raw projection stacks (tiffs), parse settings file of the scanner to produce meta data.
* `flexdata.geometry`: Geometry classes (circular, helical and linear).
* `flexdata.display`: Simple display routines for 3D arrays

Typical usage:
```python
# Import:
from flexdata import data

# Read raw projections and flat field images:
proj = data.read_tiffs(path, file_name)
flat = data.read_tiffs(path, file_name)

# Read metadata:
geom = data.read_flexraylog(path)

# Generate an ASTRA-compatible projection geometry description:
proj_geom = geom.astra_proj_geom(proj.shape)
```

## Authors and contributors

* **Alexander Kostenko** - *Initial work*
* **Allard Hendriksen** - *Packaging and installation*
* **Jan-Willem Buurlage** - *Packaging and installation*
* **Willem Jan Palenstijn** - *Packaging, installation and maintenance*
* **Alexander Skorikov** - *Packaging, installation and maintenance*

See also the list of [contributors](https://github.com/cicwi/flexdata/contributors) who participated in this project.

## How to contribute

Contributions are always welcome. If you have any issues, questions, or remarks, then please open an issue on GitHub.

## License

This project is licensed under the GNU GENERAL PUBLIC License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* To Willem Jan Palenstijn for endless advices regarding the use of ASTRA toolbox.
