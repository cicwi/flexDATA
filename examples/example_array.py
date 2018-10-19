# -*- coding: utf-8 -*-
"""
BIG array manipulations example.
"""

#%% Imports:

from flexdata import array
import numpy

#%% Initialize an array mapped on disk:

file = '[some SSD folder]/test_001'

x = numpy.memmap(file, dtype = 'float32', mode = 'w+', shape = (50,100,150))

#%% Add arrays of different dimensions to eachother (in place):

y = numpy.ones([50, 150])

array.add_dim(x, y)

print(x.min())

#%% Multiply two arrays:

y = numpy.ones([100])

array.mult_dim(x, y)

print(x.min())

#%% Slice in an arbitrary direction:

slc = array.anyslice(x, index = 89, dim = 1)

print(x[slc])

#%% Make arrays the same shape:

x = numpy.ones([100, 1, 150])
x, y = array.shape_alike(x, y)

print(x.shape)

#%% Check free memory:

print('%u GB' % array.free_memory())