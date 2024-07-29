#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This example shows how flexData can be used with ASTRA-toolbox to reconstruct the data.
You can also install flexTomo to be able to develop advanced reconstruction scripts faster.
"""

#%% Imports:

from flexdata import data
from flexdata import display
from flexdata import correct

import numpy
import astra
from pathlib import Path

#%% Read data:

path = '/ufs/ciacc/flexbox/good/'
binning = 2

dark = data.read_stack(path, 'di00', sample=binning)
flat = data.read_stack(path, 'io00', sample=binning)
proj = data.read_stack(path, 'scan_', skip=binning, sample=binning)

geom = data.parse_flexray_scansettings(path, sample=binning)

data.write_geometrytoml('./output_dir/', geom, overwrite=True)

# Apply profile correction
geom = correct.correct(geom,
                       profile='cwi-flexray-2019-04-24',
                       do_print_changes=True)
geom = correct.correct_vol_center(geom)


#%% Prepro:
flat = (flat - dark).mean(1)
proj = (proj - dark) / flat[:, None, :]
proj = -numpy.log(proj).astype('float32')
proj = numpy.ascontiguousarray(proj)

display.slice(proj, dim = 1, title = 'Projection', cmap = 'magma')

#%% Astra reconstruction:

vol = numpy.zeros([2000 // binning, 2000 // binning, 2000 // binning], dtype = 'float32')

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

display.slice(vol, dim = 0, bounds = [0, 0.04], title = 'Projection', cmap = 'magma')


# Store output data
data.write_stack(path='./output_dir/', name='slice_', data=vol)
