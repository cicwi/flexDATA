# flexDATA

This project is a part of the larger X-ray tomographic reconstruction toolbox comprised of flexDATA, flexTOMO and flexCALC.
flexDATA contains IO routines that were originally developed for the Flex-Ray X-ray CT scanner but can be used for reading and writing data from other types of CT scanners. It provides an interface between a scanner and a GPU-based CT reconstruction [ASTRA toolbox](https://www.astra-toolbox.com/). It can be especially helpful when the scanner has many degrees of freedom and the geometry of each scan is defined by a large number of parameters.

## Getting Started

It takes a few steps to setup flexDATA on your machine. We recommend that the user installs [Anaconda package manager](https://www.anaconda.com/download/) for Python 3.

### Installing with conda

Simply install with:
```
conda create -n <your-environment> python=3.6
conda install -c cicwi -c astra-toolbox/label/dev -c conda-forge -c owlas flexdata
```

### Installing from source

To install flexDATA you will need the latest version of the ASTRA toobox (preferably development version). It can be installed via command line with Anaconda:

```
conda install -c astra-toolbox/label/dev astra-toolbox
```

To install flexDATA, simply clone this GitHub project. Go to the cloned directory and run PIP installer:
```
git clone https://github.com/teascavenger/flexdata.git
cd flexdata
pip install -e .
```


## Running the examples

To learn about the functionality of the package check out our examples folder. Examples are separated into blocks that are best to run in Spyder environment step-by-step.

## Modules

flexDATA is comprised of the following modules:

* io:     read / write raw projection stacks (tiffs), parse settings file of the scanner to produce meta data.
* array:  some utility functions compatible with large arrays mapped on disk (numpy.memmap)
* scp:    wrapper around an SCP client that allows to recursevily copy folders across the network
* display:simple display routines for 3D arrays

Typical code:
```
# Import:
from flexdata import scp
from flexdata import io

# Get files from remote:
scp.ssh_get_path(local_path, remote_path, host, user)

# Read raw projections and flat field images:
proj = io.read_tiffs(path, file_name)
flat = io.read_tiffs(path, file_name)

# Read metadata:
meta = io.read_meta(path, file_type)

# Generate an ASTRA-compatible projection geometry description:
proj_geom = io.astra_proj_geom(meta['geometry'], proj.shape)
```

## Authors and contributors

* **Alexander Kostenko** - *Initial work*

See also the list of [contributors](https://github.com/cicwi/flexdata/contributors) who participated in this project.

## How to contribute

Contributions are always welcome. Please submit pull requests against the `develop` branch.

If you have any issues, questions, or remarks, then please open an issue on GitHub.

## License

This project is licensed under the GNU GENERAL PUBLIC License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* To Willem Jan Palenstijn for endles advices regarding the use of ASTRA toolbox.
