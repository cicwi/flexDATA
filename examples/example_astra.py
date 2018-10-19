#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
How to load data with flexdata and reconstruct it using ASTRA.
"""
#%% Imports:

from flexdata import io
from flexdata import array
from flexdata import display

import numpy
import astra

#%% Read data:
    
path = '/ufs/ciacc/flexbox/al_test/90KV_no_filt/'

dark = io.read_tiffs(path, 'di00')
flat = io.read_tiffs(path, 'io00')    
proj = io.read_tiffs(path, 'scan_')

meta = io.read_meta(path, 'flexray')   
 
#%% Prepro:
    
proj = (proj - dark) / (flat.mean(0) - dark)
proj = -numpy.log(proj)

proj = array.raw2astra(proj)    

display.display_slice(proj, title = 'Sinogram. What else?')

#%% Recon:

vol = numpy.zeros([50, 2000, 2000], dtype = 'float32')

# Initialize ASTRA geometries:
vol_geom = io.astra_vol_geom(meta['geometry'], vol.shape)
proj_geom = io.astra_proj_geom(meta['geometry'], proj.shape)
        
# This is ASTRAAA!!!
sin_id = astra.data3d.link('-sino', proj_geom, numpy.ascontiguousarray(proj))
vol_id = astra.data3d.link('-vol', vol_geom, numpy.ascontiguousarray(vol))

cfg = astra.astra_dict('FDK_CUDA')
cfg['ReconstructionDataId'] = vol_id
cfg['ProjectionDataId'] = sin_id

alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id, 1)
  
astra.algorithm.delete(alg_id)
astra.data3d.delete(sin_id)
astra.data3d.delete(vol_id)

#%% Display:
    
display.display_slice(vol, title = 'Volume')    
