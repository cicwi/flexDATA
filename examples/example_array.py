# -*- coding: utf-8 -*-
"""
Array manipulations example.
"""

#%% Imports:

from flexdata import array
from flexdata import display

import numpy

#%% Initialize an array mapped on disk:
file = 'C:/test/temp1/mem_001'

data_0 = numpy.memmap(file, dtype = 'float32', mode = 'w+', shape = (50,100,150))

#%% Add arrays of different dimensions (in place):
data_1 = numpy.ones([50, 150])
array.add_dim(data_0, data_1)

print(data_0.min())

#%% Multiply two arrays:
data_1 = numpy.ones([100])
array.mult_dim(data_0, data_1)

print(data_0.min())

#%% Slice in an arbitrary direction:
slc = array.anyslice(data_0, index = 89, dim = 1)

print(data_0[slc])

#%% Make arrays the same shape:
data_1 = numpy.ones([100, 1, 150])
data_0, data_1 = array.shape_alike(data_0, data_1)

print(data_0.shape)

#%% Check free memory:
print('%u GB' % array.free_memory())