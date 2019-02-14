#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test reading Flexray raw and writing ASTRA readable
"""
#%%

from flexdata import io

#%% Read / write a geometry file:

path = '/ufs/ciacc/flexbox/al_test/90KV_no_filt/'
geom = io.read_flexraylog(path, sample = 1)

print('From "scan settings.txt":')
print(geom)

io.write_toml(path + 'geometry.toml', geom)

print('Raw dictionary on disk:')
print(io.read_toml(path + 'geometry.toml'))

print('Geometry from disk:')
print(io.read_metatoml(path))

#%% Read / write raw data files:
    
dark = io.read_stack(path, 'di00')
flat = io.read_stack(path, 'io00')    
proj = io.read_stack(path, 'scan_')

io.write_stack(path, 'io_000', flat, dim = 0)    

#%% Read geometry and convert to ASTRA:

vol_geom =  geom.astra_volume_geom([100, 100, 100])
proj_geom =  geom.astra_projection_geom(proj.shape)

print('ASTRA geometries:')    
print(vol_geom)
print(proj_geom)
