#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Examples of the geometry classes usage: initialiation of a circular, helical and linear orbits; import ASTRA geometries.
"""
#%% Imports:

from flexdata import geometry
from flexdata import display

#%% Initialize a geometry record:

# Initialization:
geo = geometry.circular(src2obj = 100, det2obj = 50, det_pixel = 0.1, img_pixel = 0.05, rot_range = (0, 180), unit = 'mm')

# Print geometry parameters:
print('\n', geo)

# Additional parameters:
geo['axs_pitch'] = 30

# Plot the source orbit:
orbit = geo.get_source_orbit(50)
display.plot3d(orbit[:, 0], orbit[:, 1], orbit[:, 2], connected = True, title = 'Source orbit')

#%% Some utility functions:
prj_shape = (256, 256, 256)
vol_shape = (256, 256, 256)

print('\nDetector size and physical bounds:')
print(geo.detector_size(prj_shape))
print(geo.detector_bounds(prj_shape))

print('\nVolume size and physical bounds:')
print(geo.volume_size(vol_shape))
print(geo.volume_bounds(vol_shape))

print('\nASTRA geometry:')
print(geo.astra_projection_geom(prj_shape))
print(geo.astra_volume_geom(vol_shape))

#%% Alternative orbit shapes:

# Helix with a tilted axis:
geo = geometry.helical(src2obj = 100, det2obj =50, det_pixel = 0.1, img_pixel = 0.05, axis_range = (-50, 50), rot_range = (0, 360*3), unit = 'mm')
geo['axs_roll'] = 30

orbit = geo.get_source_orbit(50)
display.plot3d(orbit[:, 0], orbit[:, 1], orbit[:, 2], connected = True, title = 'Helical source orbit')

# Line:
geo = geometry.linear(src2obj = 100, det2obj =50, det_pixel = 0.1, img_pixel = 0.05, 
                      src_hrz_rng = (-50, 50), src_vrt_rng = (0, 0), det_hrz_rng = (50,-50), det_vrt_rng = (0,0))

orbit = geo.get_source_orbit(50)
display.plot3d(orbit[:, 0], orbit[:, 1], orbit[:, 2], connected = True, title = 'Linear source orbit')


