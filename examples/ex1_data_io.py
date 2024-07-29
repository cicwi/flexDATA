#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of some I/O utilities and geometry parsing of log files.

The dataset 'rat skull' can be found at:
    https://zenodo.org/badge/DOI/10.5281/zenodo.1164088.svg
"""
#%% Imports

from flexdata import data
from flexdata import display
from flexdata import correct
import os

#%% Read image stacks:

# Read files with 'scan_' in them (sampling = 4 for speed):
path = '/ufs/ciacc/flexbox/skull/'
proj = data.read_stack(path, 'scan_', skip = 4, sample = 4)

# Writing stack:
data.write_stack('./output_dir/binned', 'scan_', proj, dim = 1)

# Display:
display.slice(proj, dim = 0, title = 'Sinogram', cmap = 'magma')
display.slice(proj, dim = 1, title = 'Projection', cmap = 'magma')

#%% Read / write a geometry file:

# Parser for a flexray log file:
print('\nParsing "scan settings.txt":')
geom = data.parse_flexray_scansettings(path, sample = 4)
geom = correct.correct(geom, profile='cwi-flexray-2019-04-24')
geom = correct.correct_vol_center(geom)

print(geom)

# Write TOML format to disk:
data.write_geometrytoml('./output_dir', geom)

print('\nReading raw TOML:')
print(data.read_raw_toml('./output_dir/geometry.toml'))

print('\nParsing geometry.toml:')
print(data.read_geometrytoml('./output_dir', sample = 4))

