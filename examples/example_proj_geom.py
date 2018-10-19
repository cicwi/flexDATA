#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read flexray the FlexRay scanner log file and save a TOML file and an ASTRA geometry file.
"""
#%% Imports

import os
from flexdata import io

#%% Reading log file, saving ASTRA file

# Extract arguments:
path = '/ufs/ciacc/flexbox/al_test/90KV_no_filt/'

# Read:
meta = io.read_meta(path, 'flexray')   

# Write TOML:
io.write_toml(os.path.join(path, 'flexray.toml'), meta)

# Read TOML:
meta = io.read_toml(os.path.join(path, 'flexray.toml'))

# Data shape is not specified by settings file, it's the property of the projection data...
data_shape = [100,100,100]

# Create an ASTRA projection geometry file:
io.write_astra(os.path.join(path, 'projection.geom'), data_shape, meta['geometry'])

