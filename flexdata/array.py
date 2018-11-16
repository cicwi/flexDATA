#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kostenko
Created on Oct 2018

Utility functions to hande big arrays of data. All routines support memmap arrays.
However, some operations will need enough memory for one copy of the data for intermediate
results. This can be improved through better use of memmaps.
"""

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Imports >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

import numpy          # arrays arrays arrays
import psutil         # RAM test
import os             # File deletion

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class memmap(numpy.memmap):
    '''
    Standard memmaps don't seem to reliably delete files that are created on disk.
    This fixes it...
    '''     
    def delete(self):
        
        # Ref counting wit gc doesnt work.... will need to delte the file by hand.
        # self.flag[OWNDATA] doesn't work either ...
        if self.filename:
            if os.path.exists(self.filename):
            
                print('Deleting a memmap @' + self.filename)
                os.remove(self.filename)
       
def free_memory(percent = False):
    '''
    Return amount of free memory in GB.
    Args:
        percent (bool): percentage of the total or in GB.
        
    '''
    if not percent:
        return psutil.virtual_memory().available/1e9
    
    else:
        return psutil.virtual_memory().available / psutil.virtual_memory().total * 100
    
def cast2type(array, dtype, bounds = None):
    """
    Cast from float to int or float to float rescaling values if needed.
    """
    # No? Yes? OK...
    if array.dtype == dtype:
        return array
    
    # Make sure dtype is not a string:
    dtype = numpy.dtype(dtype)
    
    # If cast to float, simply cast:
    if dtype.kind == 'f':
        return array.astype(dtype)
    
    # If to integer, rescale:
    if bounds is None:
        bounds = [numpy.amin(array), numpy.amax(array)]
    
    data_max = numpy.iinfo(dtype).max
    
    array -= bounds[0]
    array *= data_max / (bounds[1] - bounds[0])
    
    array[array < 0] = 0
    array[array > data_max] = data_max
    
    new = numpy.array(array, dtype)    
    
    return rewrite_memmap(array, new)
   
def shape_alike(array_1, array_2):
    '''
    Make sure two arrays have the same shape by padding either array_1 or array_2:
        Returns: array1, array2 - reshaped.
    '''
    if array_2.ndim != array_1.ndim:
        raise Exception('Array dimensions not equal!')
        
    d_shape = numpy.array(array_2.shape)
    d_shape -= array_1.shape

    for dim in range(3):
        
        pp = d_shape[dim]
        if pp > 0:
            array_1 = pad(array_1, dim, [0, abs(pp)], mode = 'zero')
        if pp < 0:
            array_2 = pad(array_2, dim, [0, abs(pp)], mode = 'zero')
  
    return array_1, array_2

def ramp(array, dim, width, mode = 'linear'):
    """
    Create ramps at the ends of the array (without changing its size). 
    modes:
        'linear' - creates linear decay of intensity
        'edge' - smears data in a costant manner
        'zero' - sets values to zeroes
    """
    
    # Left and right:
    if numpy.size(width) > 1:
        rampl = width[0]
        rampr = width[1]
    else:
        rampl = width
        rampr = width
        
    if array.shape[dim] < (rampl + rampr):
        return array
    
    # Index of the left and right ramp:
    left_sl = anyslice(array, slice(0, rampl), dim)
    right_sl = anyslice(array, slice(-rampr, None), dim)
    
    if mode == 'zero':
        if rampl > 0:
            array[left_sl] *= 0
            
        if rampr > 0:    
            array[right_sl] *= 0
            
    elif (mode == 'edge'):
        # Set everything to the edge value:
        if rampl > 0:
            array[left_sl] *= 0
            add_dim(array[left_sl], array[anyslice(array, rampl, dim)])            
            
        if rampr > 0:    
            array[right_sl] *= 0
            add_dim(array[right_sl], array[anyslice(array, -rampr-1, dim)])            
    
    elif mode == 'linear':
        # Set to edge and multiply by a ramp:
        
        if rampl > 0:            
            # Replace values using add_dim:
            array[left_sl] *= 0
            add_dim(array[left_sl], array[anyslice(array, rampl, dim)])            
            
            mult_dim(array[left_sl], numpy.linspace(0, 1, rampl))        
            
        if rampr > 0:    
            # Replace values using add_dim:
            array[right_sl] *= 0
            add_dim(array[right_sl], array[anyslice(array, -rampr-1, dim)])            

            mult_dim(array[right_sl], numpy.linspace(1, 0, rampr))                    
        
    else:
        raise(mode, '- unknown mode! Use linear, edge or zero.')
        
    return array

def pad(array, dim, width, mode = 'edge'):
    """
    Pad an array along a given dimension.
    
    numpy.pad seems to be very memory hungry! Don't use it for large arrays.
    """
    print('Padding data...')
    
    if numpy.size(width) > 1:
        padl = width[0]
        padr = width[1]
    else:
        padl = width
        padr = width
    
    # Original shape:
    sz1 = numpy.array(array.shape)    
    sz1[dim] += padl + padr
    
    # Initialize bigger array (it's RAM-based array - need enough memory here!):
    new = numpy.zeros(sz1, dtype = array.dtype)    
    
    sl = anyslice(new, slice(padl,-padr), dim)
    
    new[sl] = array
    
    new = ramp(new, dim, width, mode)
    
    # If input is memmap - update it's size, release RAM memory.
    return rewrite_memmap(array, new)
 
def bin(array, dim = None):
    """
    Simple binning of the data along the chosen direction.
    """ 
          
    if dim is not None:
        # apply binning in one dimension
        
        # First apply division by 2:
        if (array.dtype.kind == 'i') | (array.dtype.kind == 'u'):    
            array //= 2 # important for integers
        else:
            array /= 2
            
        if dim == 0:
             array[:-1:2, :, :] += array[1::2, :, :]
             return array[:-1:2, :, :]
             
        elif dim == 1:
             array[:, :-1:2, :] += array[:, 1::2, :]
             return array[:, :-1:2, :]
             
        elif dim == 2:
             array[:, :, :-1:2] += array[:, :, 1::2]
             return array[:, :, :-1:2]
             
    else:        
    
        # First apply division by 8:
        if (array.dtype.kind == 'i') | (array.dtype.kind == 'u'):    
            array //= 8
        else:
            array /= 8
        
        # Try to avoid memory overflow here:
        for ii in range(array.shape[0]):
            array[ii, :-1:2, :] += array[ii, 1::2,:]
            array[ii, :, :-1:2] += array[ii, :,1::2]
            
        for ii in range(array.shape[2]):
            array[:-1:2,:-1:2, :, ii] += array[:-1:2,:-1:2, :, ii]
        
        return array[:-1:2, :, :]
    
def crop(array, dim, width, geometry = None):
    """
    Crop an array along the given dimension. Provide geometry if cropping the projection data,
    it will update the detector center.
    """
    if numpy.size(width) > 1:
        widthl = int(width[0])
        widthr = int(width[1])
        
    else:
        widthl = int(width) // 2
        widthr = int(width) - widthl 
   
    # Geometry shifts:
    h = 0
    v = 0
    
    # If widthr we need to sample up to None index according to Python rules
    widthr = -widthr
    
    if dim == 0:
        v = (widthl + widthr)
        
        if widthr == 0: widthr = None
        new = array[widthl:widthr, :,:]
        
    elif dim == 1:
        h = (widthl + widthr)
        
        if widthr == 0: widthr = None
        new = array[:,widthl:widthr,:]
        
    elif dim == 2:
        h = (widthl + widthr)
        
        if widthr == 0: widthr = None
        new = array[:,:,widthl:widthr]  
    
    if geometry: shift_geometry(geometry, h/2, v/2)
        
    # Its better to leave the memmap file as it is. Return a view to it:
    return new

def shift_geometry(geometry, hrz, vrt, update_volume_pos = True):
    """
    Apply geometry shift in pixels.
    """    
    hrz = hrz * geometry['det_pixel']
    vrt = vrt * geometry['det_pixel']
    
    geometry['det_hrz'] += hrz
    geometry['det_vrt'] += vrt
    
    if update_volume_pos:        
        # Here we are computing magnification without taking into account vol_tra[1], det_mag
        m = (geometry['src2obj'] + geometry['det2obj']) / geometry['src2obj']
        geometry['vol_tra'][2] += hrz / m
        geometry['vol_tra'][0] += vrt / m    

def raw2astra(array):
    """
    Convert a given numpy array (sorted: index, hor, vert) to ASTRA-compatible projections stack
    """    
    # Transpose seems to be compatible with memmaps:    
    array = numpy.transpose(array, [1,0,2])
    array = numpy.flipud(array)
        
    return array

def medipix2astra(array):
    """
    Convert a given numpy array (sorted: index, hor, vert) to ASTRA-compatible projections stack
    """    
    # Don't apply ascontignuousarray on memmaps!    
    array = numpy.transpose(array, [2,0,1])
    array = numpy.flipud(array)
    
    return array

def rewrite_memmap(old_array, new_array):
    '''
    Reshaping memmaps is tough. We will recreate one instead hoping that this will not overflow our RAM...
    This is a dirty qick fix! Try to use resize instead!
    '''
    if isinstance(old_array, memmap):
        
        # Sometimes memmaps are created without a file (a guess they are kind of copies of views of actual memmaps...)
        if old_array.filename:
            # Trick is to open the file in r+ mode:
            old_array = memmap(old_array.filename, dtype='float32', mode = 'r+', shape = new_array.shape)
            old_array[:] = new_array[:]
            
        else:
            old_array = new_array
    else:
        del old_array
        
        # array is not a memmmap:
        old_array = new_array
        
    return old_array


def add_dim(array_1, array_2):
    """
    Add two arrays with arbitrary dimensions. We assume that one or two dimensions match.
    """
    
    # Shapes to compare:
    shp1 = numpy.shape(array_1)
    shp2 = numpy.shape(array_2)
    
    dim1 = numpy.ndim(array_1)
    dim2 = numpy.ndim(array_2)
    
    if dim1 - dim2 == 1:
        
        # Find dimension that is missing in array_2:
        dim = [ii not in shp2 for ii in shp1].index(True)
        
        if dim == 0:
            array_1 += array_2[None, :, :]
        elif dim == 1:
            array_1 += array_2[:, None, :]
        elif dim == 2:
            array_1 += array_2[:, :, None]            
        
    elif dim1 - dim2 == 2:
        # Find dimension that is matching in array_2:
        dim = [ii in shp2 for ii in shp1].index(True)
        
        if dim == 0:
            array_1 += array_2[:, None, None]
        elif dim == 1:
            array_1 += array_2[None, :, None]
        else:
            array_1 += array_2[None, None, :]
            
    else:
        raise Exception('ERROR! array_1.ndim - array_2.ndim should be 1 or 2')
           
def mult_dim(array_1, array_2):    
    """
    Multiply a 3D array by a 1D or a 2D vector along one of the dimensions.
    """
    # Shapes to compare:
    shp1 = numpy.shape(array_1)
    shp2 = numpy.shape(array_2)
    
    dim1 = numpy.ndim(array_1)
    dim2 = numpy.ndim(array_2)
    
    if dim1 - dim2 == 1:
        
        # Find dimension that is missing in array_2:
        dim = [ii not in shp2 for ii in shp1].index(True)
        
        if dim == 0:
            array_1 *= array_2[None, :, :]
        elif dim == 1:
            array_1 *= array_2[:, None, :]
        elif dim == 2:
            array_1 *= array_2[:, :, None]            
        
    elif dim1 - dim2 == 2:
        
        # Find dimension that is matching in array_2:
        dim = [ii in shp2 for ii in shp1].index(True)
        
        if dim == 0:
            array_1 *= array_2[:, None, None]
        elif dim == 1:
            array_1 *= array_2[None, :, None]
        else:
            array_1 *= array_2[None, None, :]
            
    else:
        raise('ERROR! array_1.ndim - array_2.ndim should be 1 or 2')
        
def anyslice(array, index, dim):
    """
    Slice an array along an arbitrary dimension.
    """
    sl = [slice(None)] * array.ndim
    sl[dim] = index
    
    # Nowadays python asks for tuples:
    sl = tuple(sl)
      
    return sl   

def detector_size(shape, geometry):
    '''
    Get the size of detector in mm.
    '''       
    return geometry['det_pixel'] * numpy.array(shape)

def volume_bounds(proj_shape, geometry):
    '''
    A very simplified version of volume bounds...
    '''
    # TODO: Compute this propoerly.... Dont trust the horizontal bounds!!!
    
    # Detector bounds:
    det_bounds = detector_bounds(proj_shape, geometry)
    
    # Demagnify detector bounds:
    fact = geometry['src2obj'] / (geometry['src2obj'] + geometry['det2obj'])
    vrt = numpy.array(det_bounds['vrt'])
    vrt_bounds = (vrt * fact + geometry['src_vrt'] * (1 - fact))
    
    hrz = numpy.array(det_bounds['hrz'])
    max_x = max(hrz - geometry['axs_hrz'])
    
    hrz_bounds = [geometry['vol_tra'][2] - max_x, geometry['vol_tra'][2] + max_x]
    mag_bounds = [geometry['vol_tra'][1] - max_x, geometry['vol_tra'][1] + max_x]
            
    vol_bounds = {'vrt':numpy.array(vrt_bounds), 
                  'mag': numpy.array(mag_bounds), 
                  'hrz': numpy.array(hrz_bounds)}
    
    return vol_bounds

def volume_shape(proj_shape, geometry):
    '''
    Based on physical volume bnounds compute shape in pixels:
    '''
    bounds = volume_bounds(proj_shape, geometry)

    range_vrt = numpy.ceil(bounds['vrt'] / geometry['img_pixel'])
    range_hrz = numpy.ceil(bounds['hrz'] / geometry['img_pixel'])
    range_mag = numpy.ceil(bounds['mag'] / geometry['img_pixel'])
    
    range_vrt = range_vrt[1] - range_vrt[0]
    range_hrz = range_hrz[1] - range_hrz[0]
    range_mag = range_mag[1] - range_mag[0]
    
    return numpy.int32([range_vrt, range_mag, range_hrz])
    
def detector_bounds(shape, geometry):
    '''
    Get the boundaries of the detector in mm
    '''   
    bounds = {}

    xmin = geometry['det_hrz'] - geometry['det_pixel'] * shape[2] / 2
    xmax = geometry['det_hrz'] + geometry['det_pixel'] * shape[2] / 2

    ymin = geometry['det_vrt'] - geometry['det_pixel'] * shape[0] / 2
    ymax = geometry['det_vrt'] + geometry['det_pixel'] * shape[0] / 2

    bounds['hrz'] = [xmin, xmax]
    bounds['vrt'] = [ymin, ymax]
    
    return bounds
    
def tiles_shape(shape, geometry_list):
    """
    Compute the size of the stiched dataset.
    Args:
        shape: shape of a single projection stack.
        geometry_list: list of geometries.
        
    """
    # Phisical detector size:
    min_x, min_y = numpy.inf, numpy.inf
    max_x, max_y = -numpy.inf, -numpy.inf
    
    det_pixel = geometry_list[0]['det_pixel']
    
    # Find out the size required for the final dataset
    for geo in geometry_list:
        
        bounds = detector_bounds(shape, geo)
        
        min_x = min([min_x, bounds['hrz'][0]])
        min_y = min([min_y, bounds['vrt'][0]])
        max_x = max([max_x, bounds['hrz'][1]])
        max_y = max([max_y, bounds['vrt'][1]])
        
    # Big slice:
    new_shape = numpy.array([(max_y - min_y) / det_pixel, shape[1], (max_x - min_x) / det_pixel])                     
    new_shape = numpy.round(new_shape).astype('int')
    
    # Copy one of the geometry records and sett the correct translation:
    geometry = geometry_list[0].copy()
    
    geometry['det_hrz'] = (max_x + min_x) / 2
    geometry['det_vrt'] = (max_y + min_y) / 2
    
    # Update volume center:
    #geometry['vol_vrt'] = (geometry['det_vrt'] * geometry['src2obj'] + geometry['src_vrt'] * geometry['det2obj']) / geometry.get('src2det')
    #geometry['vol_hrz'] = (geometry['det_hrz'] + geometry['src_hrz']) / 2
    geometry['vol_tra'][0] = (geometry['det_vrt'] * geometry['src2obj'] + geometry['src_vrt'] * geometry['det2obj']) / geometry.get('src2det')
    #geometry['vol_tra'][2] = (geometry['det_hrz'] + geometry['src_hrz']) / 2
    geometry['vol_tra'][2] = geometry['axs_hrz']

    return new_shape, geometry
