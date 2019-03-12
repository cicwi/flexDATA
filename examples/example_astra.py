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
    
path = 'D:\data\skull'

dark = io.read_stack(path, 'di00', sample= 4)
flat = io.read_stack(path, 'io00', sample= 4)    
proj = io.read_stack(path, 'scan_', skip = 4, sample= 4)

geom = io.read_flexraylog(path)   
 
#%% Prepro:
   
proj = (proj - dark) / (flat.mean(0) - dark)
proj = -numpy.log(proj)

proj = array.raw2astra(proj)    
proj = numpy.ascontiguousarray(proj.astype('float32')) 

display.slice(proj, title = 'Sinogram. What else?')

#%% Recon:

vol = numpy.zeros([50, 200, 200], dtype = 'float32') + 100

# Initialize ASTRA geometries:
vol_geom = geom.astra_volume_geom(vol.shape)
proj_geom = geom.astra_projection_geom(proj.shape)
        
# This is ASTRAAA!!!
sin_id = astra.data3d.link('-sino', proj_geom, proj)
vol_id = astra.data3d.link('-vol', vol_geom, vol)

cfg = astra.astra_dict('FDK_CUDA')
cfg['ReconstructionDataId'] = vol_id
cfg['ProjectionDataId'] = sin_id

alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id, 1)
  
astra.algorithm.delete(alg_id)
astra.data3d.delete(sin_id)
astra.data3d.delete(vol_id)

#%% Display:
    
display.slice(vol, title = 'Volume')    
