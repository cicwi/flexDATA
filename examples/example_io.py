#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test reading Flexray raw and writing ASTRA readable
"""
#%%
import flexdata as data

#%% Read / write a geometry file:

path = '/ufs/ciacc/flexbox/al_test/90KV_no_filt/'

meta = data.io.read_meta(path, 'flexray')

data.io.write_toml(path + 'flexray.toml', meta)

#%% Read / write raw data files:
    
dark = data.io.read_tiffs(path, 'di')
flat = data.io.read_tiffs(path, 'io')    
proj = data.io.read_tiffs(path, 'scan_')

#%% Read geometry and convert to ASTRA:

meta_1 = data.io.read_toml(path + 'flexray.toml') 

vol_geom =  data.io.astra_vol_geom(meta['geometry'], [100, 100, 100])
proj_geom =  data.io.astra_proj_geom(meta['geometry'], proj.shape)
    
print(vol_geom)
print(proj_geom)
