#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read flexray log file and save a TOML file and an ASTRA geometry file.
"""
import os
from flexdata import io

# Extract arguments:
path = '/ufs/ciacc/flexbox/al_test/90KV_no_filt/'

print('Reading log file at:', path)

# Read:
meta = io.read_meta(path, 'flexray')   

# Write:
io.write_toml(os.path.join(path, 'flexray.toml'), meta)

data_shape = [100,100,100]
io.write_astra(os.path.join(path, 'projection.geom'), data_shape, meta['geometry'])

