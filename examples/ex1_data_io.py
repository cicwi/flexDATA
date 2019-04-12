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

#%% Read image stacks:

# Read files with 'scan_' in them (sampling = 4 for speed):
path = '/ufs/ciacc/flexbox/skull/'
proj = data.read_stack(path, 'scan_', skip = 4, sample = 4)

# Writing stack:
data.write_stack(path + 'binned', 'scan_', proj, dim = 1)

# Display:
display.slice(proj, dim = 0, title = 'Sinogram', cmap = 'magma')
display.slice(proj, dim = 1, title = 'Projection', cmap = 'magma')

#%% Read / write a geometry file:

# Parcer for a flexray log file:
geom = data.read_flexraylog(path, sample = 4)

print('\nParsing "scan settings.txt":')
print(geom)

# Write TOML format to disk:
data.write_toml(path + 'geometry.toml', geom)

print('\nReading raw TOML:')
print(data.read_toml(path + 'geometry.toml'))

print('\nParsing raw TOML:')
print(data.read_geometry(path, sample = 4))

