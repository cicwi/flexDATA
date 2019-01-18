#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kostenko
Created on Oct 2018

This module will contain read / write routines to convert FlexRay scanner data into ASTRA compatible data

We can now read/write:
    image files (tiff stacks)
    log files from Flex ray (settings.txt)
    toml geometry files (metadata.toml)

We can also copy data over SCP!

"""

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Imports >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

import astra          # The Mother-ASTRA
import numpy          # arrays arrays arrays
import os             # operations with filenames
import re             # findall function
import warnings       # warn me if we are in trouble!
import imageio        # io for images
import psutil         # RAM tester
import toml           # TOML format parcer
import transforms3d   # rotation matrices
from tqdm import tqdm # progress bar
import time           # pausing

from scipy.io import loadmat # Reading matlab format
from . import array          # operations with arrays

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Constants >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Geometry types:
GEOM_SIMPLE = 'simple'
GEOM_STAOFF = 'static_offsets'
GEOM_LINOFF = 'linear_offsets'

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 
def init_meta(geometry = None):
    """
    Initialize the meta record (contains: geometry, settings and description).
    """
    
    # Settings and description:
    settings = {'voltage': 0,
                'power': 0,
                'averages': 0,
                'mode':'n/a',
                'filter':'n/a',
                'exposure':0}
    
    description = { 'name':'n/a',
                    'comments':'n/a',
                    'date':'n/a',
                    'duration':'n/a'
                    }
        
    # Geometry initialization:
    if not geometry:    
        geometry = init_geometry()     
                            
    return {'geometry':geometry, 'settings':settings, 'description':description}

def init_geometry(src2obj = 0, det2obj = 0, det_pixel = 0, unit = 'millimetre', theta_range = [0, 360], geom_type = 'simple'):
    """
    Initialize the geometry record with a basic geometry records.
    
    Args:
        src2obj     : source to object distance
        det2obj     : detector to object distance
        det_pixel   : detector pixel size
        unit        : metric unit
        theta_range : range of rotation
        geom_type   : can be 'simple', 'static_offsets' or 'linear_offsets'
        
    Returns: 
        geometry: a dictionary containing records of one of three types of the geometry.
    """
    if src2obj != 0:
        img_pixel = det_pixel / ((src2obj + det2obj) / src2obj)
        
    else:
        img_pixel = 0
        
    # Create a geometry dictionary:
    geometry = {'type':geom_type, 
                'unit':unit, 
                'det_pixel':det_pixel, 
                'src2obj': src2obj, 
                'det2obj':det2obj, 
                'theta_min': theta_range[0], 
                'theta_max': theta_range[1],
                
                'img_pixel':img_pixel, 
                'vol_sample':[1, 1, 1], 
                'proj_sample':[1, 1, 1],
                
                'axs_hrz':0,
                              
                'vol_rot':[0.,0.,0.], 
                'vol_tra':[0.,0.,0.]
                }
    
    # If type is not 'simple', populate with additional records:
    if geom_type == GEOM_STAOFF:
        # Source position:
        geometry['src_vrt'] = 0.
        geometry['src_hrz'] = 0.
        geometry['src_mag'] = 0. # This value should most of the times be zero since the SOD and SDD distances are known
        
        # Detector position:        
        geometry['det_vrt'] = 0.
        geometry['det_hrz'] = 0.
        geometry['det_mag'] = 0. # same here
        geometry['det_roll'] = 0.
        geometry['det_roll'] = 0.
        geometry['det_pitch'] = 0.
        geometry['det_yaw'] = 0.
        
        # Axis of rotation position:        
        geometry['axs_hrz'] = 0.
        geometry['axs_mag'] = 0. # same here
        geometry['axs_rot'] = numpy.zeros(2, dtype = 'float32')
        
        
    if geom_type == GEOM_LINOFF:
        
        zz = numpy.zeros(2, dtype = 'float32')
        
        # Source position:
        geometry['src_vrt'] = zz
        geometry['src_hrz'] = zz
        geometry['src_mag'] = zz 
        
        # Detector position:      
        geometry['det_vrt'] = zz
        geometry['det_hrz'] = zz
        geometry['det_mag'] = zz # same here
        geometry['det_roll'] = zz
        geometry['det_pitch'] = zz
        geometry['det_yaw'] = zz

        # Axis of rotation position:        
        geometry['axs_hrz'] = zz
        geometry['axs_mag'] = zz # same here 
        geometry['axs_rot'] = numpy.zeros((2,2), dtype = 'float32')
        
    return geometry

def read_flexray(path, sample = 1, skip = 1, memmap = None, proj_number = None):
    '''
    Read raw projecitions, dark and flat-field, scan parameters from a typical FlexRay folder.
    
    Args:
        path   (str): path to flexray data.
        skip   (int): read every ## image
        sample (int): keep every ## x ## pixel
        memmap (str): output a memmap array using the given path
        proj_number (int): force projection number (treat lesser numbers as missing)
        
    Returns:
        proj (numpy.array): projections stack
        flat (numpy.array): reference flat field images
        dark (numpy.array): dark field images   
        meta (dict): description of the geometry, physical settings and comments
        
    '''
    
    dark = read_tiffs(path, 'di00', skip, sample)
    flat = read_tiffs(path, 'io00', skip, sample)
    
    # Read the raw data
    # check whether files were actually read
    if proj_number:
        success = numpy.zeros(int(numpy.ceil(proj_number / skip)))        
    else:
        success = []
        
    proj = read_tiffs(path, 'scan_', skip, sample, [], [], 'float32', memmap = memmap, success = success)
    
    # Try to retrieve metadata:
    try:
        
        meta = read_meta(path, 'metadata', sample)   
        
    except:
        
        meta = read_meta(path, 'flexray', sample)  
        
    # Check success. If a few files were not read - interpolate, otherwise adjust the meta record.
    proj = _check_success_(proj, meta, success)
    
    return proj, flat, dark, meta
    
def read_tiffs(path, name, skip = 1, sample = 1, x_roi = [], y_roi = [], dtype = None, flipdim = False, memmap = None, success = None):
    """
    Read tiff files stack and return a numpy array.
    
    Args:
        path (str): path to the files location
        name (str): common part of the files name
        skip (int): read every so many files
        sample (int): sampling factor in x/y direction
        x_roi ([x0, x1]): horizontal range
        y_roi ([y0, y1]): vertical range
        dtype (str or numpy.dtype): data type to return
        flipdim (bool): apply dimension switch for ASTRA compatibility
        memmap (str): if provided, return a disk mapped array to save RAM
        success(array): map of the files that could be loaded (equals 0 in case of a read failure)
        
    Returns:
        numpy.array : 3D array with the first dimension representing the image index
        
    """  
    return read_stack(path, name, skip = skip, sample = sample, x_roi = x_roi, y_roi = y_roi, shape = [], 
               dtype = dtype, format = 'tiff', flipdim = flipdim, memmap = memmap, success = success)
        
def read_stack(path, name, skip = 1, sample = 1, x_roi = [], y_roi = [], shape = None, 
               dtype = None, format = None, flipdim = False, memmap = None, success = None):    
    """
    Read stack of files and return a numpy array.
    
    Args:
        path (str): path to the files location
        name (str): common part of the files name
        skip (int): read every so many files
        sample (int): sampling factor in x/y direction
        x_roi ([x0, x1]): horizontal range
        y_roi ([y0, y1]): vertical range
        shape (array): shape of the files. Use it when the format is 'raw'.
        dtype (str or numpy.dtype): data type of the files
        format (str): file format ('tif', 'raw', etc)
        flipdim (bool): apply dimension switch for ASTRA compatibility
        memmap (str): if provided, return a disk mapped array to save RAM
        success(array): map of the files that could be loaded (equals 0 in case of a read failure)
        
    Returns:
        numpy.array : 3D array with the first dimension representing the image index
        
    """  
    # Retrieve file names, sorted by name
    files = get_files_sorted(path, name)    
    
    if len(files) == 0: raise IOError('Tiff files not found at:', os.path.join(path, name))
    
    # Apply skip:
    files = files[::skip]
    
    # Confirm the shape and the dtype of the image:
    image = read_image(files[0], sample, x_roi, y_roi, shape, format, dtype)
    
    shape = list(image.shape)
    shape[0] *= sample
    shape[1] *= sample
    
    dtype = image.dtype
        
    file_n = len(files)
    proj_n = len(files) # may change if projection_number was forced
    
    # Initialize sucess index:
    if success is not None:
        if len(success) == 0:
            # this will be visible outside:
            success[:] = numpy.zeros(file_n)
        else:
            
            success *= 0
            
            if len(success) > file_n:
                warnings.warn('The file number is not equal to the forced projection number! Hopefully everything will be fine...')
                proj_n = len(success)
        
    # Create a mapped array if needed:
    #if shape and sample are used, shape means the shape of the image on disk before subsampling
    shape_samp = (proj_n, shape[0] // sample, shape[1] // sample)
    
    if memmap:
        data = array.memmap(memmap, dtype=dtype, mode='w+', shape = shape_samp)
        
    else:    
        data = numpy.zeros(shape_samp, dtype = dtype)
    
    # In flexbox this function can handle tiff stacks with corrupted files. 
    # Here I removed this functionality to make code simplier.
        
    time.sleep(0.3) # This is needed to let print message be printed before the next porogress bar is created
            
    # Loop with a progress bar:
    for k in tqdm(range(len(files)), unit = 'files'):
        
        # Use try...escept only is success array is provided. Otherwise, crash on errors
        if success is not None:
            try:
                im = read_image(files[k], sample, x_roi, y_roi, shape, format, dtype)
                data[k, :, :] = im     
                success[k] = 1
                
            except Exception:
                
                success[k] = 0            
                warnings.warn('Error reading file:' + files[k])
                pass
        else:
            im = read_image(files[k], sample, x_roi, y_roi, shape, format, dtype)    

            data[k, :, :] = im
 
    # Get rid of the corrupted data:
    if success is not None:                
        if sum(success) != file_n:
            warnings.warn('%u files are CORRUPTED!'%(file_n - sum(success)))
                    
        print('%u files were loaded. %u%% memory left (%u GB).' % (sum(success), free_memory(True), free_memory(False)))
    else:
        print('%u files were loaded. %u%% memory left (%u GB).' % (len(files), free_memory(True), free_memory(False)))
    
    # Apply dimension switch:    
    if flipdim:    
        data = array.raw2astra(data)    
        
    time.sleep(0.3) # This is needed to let print message be printed before the next porogress bar is created
    
    return data

def write_tiffs(path, name, data, dim = 1, skip = 1, dtype = None, zip = False):
    """
    Write a tiff stack.
    
    Args:
        path (str): destination path
        name (str): first part of the files name
        data (numpy.array): data to write
        dim (int): dimension along which array is separated into images
        skip (int): how many images to skip in between
        dtype (type): forse this data type   
        zip (bool): use loseless zip compression or not.
    """
    write_stack(path, name, data, dim = dim, skip = skip, dtype = dtype, zip = zip, format = 'tiff')
                 
def write_stack(path, name, data, dim = 1, skip = 1, dtype = None, zip = False, format = 'tiff'):
    """
    Write an image stack.
    
    Args:
        path (str): destination path
        name (str): first part of the files name
        data (numpy.array): data to write
        dim (int): dimension along which array is separated into images
        skip (int): how many images to skip in between
        dtype (type): forse this data type   
        compress (str): use None, 'zip' or 'jp2'.
        format (str): file extension ('raw', 'tiff', 'jp2', etc)
    """
    
    print('Writing data...')
    
    # Add underscore:
    if '_' not in name:
        name = name + '_'
    
    # Make path if does not exist:
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Write files stack:    
    file_num = int(numpy.ceil(data.shape[dim] / skip))

    bounds = [data.min(), data.max()]
    
    # To let things be printed in time:
    time.sleep(0.5)                            
    
    for ii in tqdm(range(file_num), unit = 'file'):
        
        path_name = os.path.join(path, name + '%06u'% (ii*skip))
        
        # Extract one slice from the big array
        sl = array.anyslice(data, ii * skip, dim)
        img = data[sl]
          
        # Cast data to another type if needed
        if dtype is not None:
            img = array.cast2type(img, dtype, bounds)
        
        # Write it!!!
        if format == 'raw':
            img.tofile(os.path.join(path_name, '.raw'))
            
        else:            
            if zip:
                write_image(path_name + '.' + format, img, 1)
                
            else:
                write_image(path_name + '.' + format, img, 0)   
                
    
def write_image(filename, image, compress = 0):
    """
    Write a single image. Use compression if needed (0-9).
    """     
    with imageio.get_writer(filename) as w:
        w.append_data(image, {'compress': compress})

def read_image(file, sample = 1, x_roi = [], y_roi = [], shape = None, format = None, dtype = None):
    """
    Read a single image. Use sampling and roi parameters to reduce the array size. 
    Use shape, format and dtype to force file reading settings.
    
    """
    
    # File header size:
    header = 0
    
    # File extension:
    ext = os.path.splitext(file)[1]
    
    if (format == 'raw') | (ext == '.raw'):
        if not dtype:
            raise Exception('Define a dtype when reading "raw" format.')
                
        # First file in the stack:    
        im = numpy.fromfile(file, dtype)
        
        if not shape:
            sz = numpy.sqrt(im.size)
            raise Exception('Define a shape when reading "raw" format. Should be ~ (%0.2f, %0.2f)' % (sz,sz))
        
        # Size of the intended array:
        sz = numpy.prod(shape)
        
        # In raw formats header may be encountered. We will estimate it automatically:
        header = (im.size - sz)
         
        if header < 0:
            raise Exception('Image size %u is smaller than the declared size %u' % (im.size, sz))
            
        #if header > 0:
        #    print('WARNING: file size is larger than the given shape. Assuming header length: %u' % header)
         
        # 1D -> 2D
        im = im[header:]    
        im = im.reshape(shape)
        
    elif ext == '':
        
        # FIle has no extension = use the one defined by the user.
        if not format: 
            raise Exception("Can't find extension of the file. Use format to provide file format.")
            
        im = imageio.imread(file, format = format)
        
    elif ext == '.mat':
        
        # Read matlab file:
        dic = loadmat(file)
        
        # We will assume that there is a single variable in this mat file:
        var_key = [key for key in dic.keys() if not '__' in key][0]
        im = dic[var_key]
        
    else:
        # Files with normal externsions:
        im = imageio.imread(file)
        
    if dtype:
        im = im.astype(dtype)
        
    # Sum RGB    
    if im.ndim > 2:
       im = im.mean(2)
         
    # TODO: Use kwags offset and size to apply roi!
    if (y_roi != []):
        im = im[y_roi[0]:y_roi[1], :]
    if (x_roi != []):
        im = im[:, x_roi[0]:x_roi[1]]

    im = _sample_image_(im, sample)
    
    return im

def read_meta(path, meta_type = 'flexray', sample = 1):
    """
    Read the log file and return dictionaries with parameters of the scan.
    
    Args:
        path (str): path to the files location
        meta_type (str): type of the meta file: 'flexray' (read settings file), 'metadata' (meta script output) or 'toml' (raw meta record saved in toml format).
        
    Returns:    
        geometry : src2obj, det2obj, det_pixel, thetas, det_hrz, det_vrt, det_mag, det_roll, src_hrz, src_vrt, src_mag, axs_hrz, vol_hrz, vol_tra 
        settings : physical settings - voltage, current, exposure
        description : lyrical description of the data
    """
    
    if meta_type == 'flexray': 
        
        # Read file and translate:
        records = _file_to_dictionary_(path, 'settings.txt', separator = ':')
        meta = _flexray_translate_(records)
        
    elif meta_type == 'metadata': 
        
        # Read file and translate:
        records = _file_to_dictionary_(path, 'metadata.toml', separator = '=')
        meta = _metadata_translate_(records)
        
    elif meta_type == 'toml':
        
        meta = read_toml(os.path.join(path, 'meta.toml'))
            
    else:
        raise ValueError('Unknown meta_type: ' + meta_type)
        
    # Apply external sampling to the pixel sizes if needed:
    meta['geometry']['det_pixel'] *= sample
    meta['geometry']['img_pixel'] *= sample    
      
    # Convert units to standard:    
    unit_to_mm(meta)    
        
    # Check if all th relevant fields are there:
    _sanity_check_(meta)
    
    return meta

def unit_to_mm(meta):
    '''
    Converts a meta record to standard units (mm).
    '''
    
    try:
        unit = _parse_unit_(meta['geometry']['unit'])
        
        meta['geometry']['det_pixel'] *= unit
        meta['geometry']['src2obj'] *= unit
        meta['geometry']['det2obj'] *= unit
        
        meta['geometry']['src_vrt'] *= unit
        meta['geometry']['src_hrz'] *= unit
        meta['geometry']['src_mag'] *= unit
        
        meta['geometry']['det_vrt'] *= unit
        meta['geometry']['det_hrz'] *= unit
        meta['geometry']['det_mag'] *= unit
        
        meta['geometry']['axs_hrz'] *= unit
        meta['geometry']['axs_mag'] *= unit
        
        meta['geometry']['unit'] = 'millimetre'
        
        meta['geometry']['img_pixel'] *= unit
        meta['geometry']['vol_tra'] *= unit

    except:
        print('Faulty geoemtry record')
        print(meta['geometry'])
        raise Exception('Unit conversion failed')
       
def read_toml(file_path):
    """
    Read a toml file.
    """  
    # TOML is terrible sometimes... which is why we are doing some fixing of the file that is loaded before parsing
    file = open(file_path,'r') 
    
    s = file.read()
    file.close()
    
    s = s.replace(u'\ufeff', '')
    s = s.replace('(', '[')
    s = s.replace(')', ']')
        
    meta = toml.loads(s)
    
    # Somehow TOML doesnt support numpy. Here is a workaround:
    for key in meta.keys():
        if isinstance(meta[key], dict):
            for subkey in meta[key].keys():
                meta[key][subkey] = _python2numpy_(meta[key][subkey])
        else:
            meta[key] = _python2numpy_(meta[key])

    return meta        
        
def write_toml(filename, meta):
    """
    Write a toml file.
    """
        
    # Make path if does not exist:
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)

    # It looks like TOML module doesnt like numpy arrays and numpy types. 
    # Use lists and native types for TOML.
    for key in meta.keys():
        if isinstance(meta[key], dict):
            for subkey in meta[key].keys():
                meta[key][subkey] = _numpy2python_(meta[key][subkey])
        else:
            meta[key] = _numpy2python_(meta[key])
            
    # Save TOML to a file:
    with open(filename, 'w') as f:
        d = toml.dumps(meta)
        f.write(d)
        
        #toml.dump(meta, f)
        
def _numpy2python_(numpy_var):
    """
    Small utility to translate numpy to standard python (needed for TOML compatibility)
    """        
    # TOML parcer doesnt like tuples:
    if isinstance(numpy_var, tuple):
        numpy_var = list(numpy.round(numpy_var, 6))
    
    # Numpy array:
    if isinstance(numpy_var, numpy.ndarray):
        numpy_var = numpy.round(numpy_var, 6).tolist()
    
    # Numpy scalar:
    if isinstance(numpy_var, numpy.generic):
        numpy_var = numpy.round(numpy_var, 6).item()
    
    # If list still use round:
    if isinstance(numpy_var, list):
        for ii in range(len(numpy_var)):
            if type(numpy_var[ii]) == 'float':
                numpy_var[ii] = numpy.round(numpy_var[ii], 6)
        
    return numpy_var
    
def _python2numpy_(var):
    """
    Small utility to translate standard python to numpy (needed for TOML compatibility)
    """        
    # Numpy array:
    if isinstance(var, list):
        var = numpy.array(var, type(var[0]))

    return var
        
def write_astra(filename, data_shape, meta):
    """
    Write an astra-readable projection geometry vector.
    """        
    geom = astra_proj_geom(meta, data_shape)
    
    # Make path if does not exist:
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)
    
    numpy.savetxt(filename, geom['Vectors']) 
    
def astra_vol_geom(geometry, vol_shape, slice_first = None, slice_last = None):
    '''
    Initialize ASTRA volume geometry.        
    '''
    # Shape and size (mm) of the volume
    vol_shape = numpy.array(vol_shape)
        
    # Use 'img_pixel' to override the voxel size:
    sample =  geometry.get('vol_sample')   
    voxel = numpy.array(sample) * geometry.get('img_pixel')

    size = vol_shape * voxel

    if (slice_first is not None) & (slice_last is not None):
        # Generate volume geometry for one chunk of data:
                   
        length = vol_shape[0]
        
        # Compute offset from the centre:
        centre = (length - 1) / 2
        offset = (slice_first + slice_last) / 2 - centre
        offset = offset * voxel[0]
        
        shape = [slice_last - slice_first + 1, vol_shape[1], vol_shape[2]]
        size = shape * voxel[0]

    else:
        shape = vol_shape
        offset = 0     
        
    #vol_geom = astra.creators.create_vol_geom(shape[1], shape[2], shape[0], 
    vol_geom = astra.create_vol_geom(shape[1], shape[2], shape[0], 
              -size[2]/2, size[2]/2, -size[1]/2, size[1]/2, 
              -size[0]/2 + offset, size[0]/2 + offset)
        
    return vol_geom   
    
def astra_proj_geom(geometry, data_shape, index = None):
    """
    Generate the vector that describes positions of the source and detector.
    Works with three types of geometry: simple, static_offsets, linear_offsets.
    
    Args:
        geometry  : geometry record of one of three types
        data_shape: [detector_count_z, theta_count, detector_count_x]
        index     : if provided - sequence of the rotation angles
    """   
    
    # Basic geometry:
    det_count_x = data_shape[2]
    det_count_z = data_shape[0]
    theta_count = data_shape[1]

    det_pixel = geometry['det_pixel'] * numpy.array(geometry.get('proj_sample'))
    
    src2obj = geometry['src2obj']
    det2obj = geometry['det2obj']

    # Check if _thetas_ are defined explicitly:
    if geometry.get('_thetas_') is not None:
        thetas = geometry['_thetas_'] / 180 * numpy.pi
        
        if len(thetas) != theta_count: 
            raise IndexError('Length of the _thetas_ array doesn`t match withthe number of projections: %u v.s. %u' % (len(thetas), theta_count))
    else:
        
        thetas = numpy.linspace(geometry.get('theta_min'), geometry.get('theta_max'),theta_count, dtype = 'float32') / 180 * numpy.pi

    # Inintialize ASTRA projection geometry to import vector from it
    if (index is not None):
        
        thetas = thetas[index]
       
    proj_geom = astra.create_proj_geom('cone', det_pixel[1], det_pixel[0], det_count_z, det_count_x, thetas, src2obj, det2obj)
    
    # Modify proj_geom if geometry is of type: static_offsets or linear_offsets:
    proj_geom = _modify_astra_vector_(proj_geom, geometry)
    
    return proj_geom
    
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
    
def pixel2mm(value, geometry):
    """
    Convert pixels to millimetres by multiplying the value by img_pixel 
    """
    m = (geometry['src2obj'] + geometry['det2obj']) / geometry['src2obj']
    img_pixel = geometry['det_pixel'] / m

    return value * img_pixel
      
def mm2pixel(value, geometry):
    """
    Convert millimetres to pixels by dividing the value by img_pixel 
    """
    m = (geometry['src2obj'] + geometry['det2obj']) / geometry['src2obj']
    img_pixel = geometry['det_pixel'] / m

    return value / img_pixel

def get_files_sorted(path, name):
    """
    Sort file entries using the natural (human) sorting
    """
    # Get the files
    files = os.listdir(path)
    
    # Get the files that are alike and sort:
    files = [os.path.join(path,x) for x in files if (name in x)]

    # Keys
    keys = [int(re.findall('\d+', f)[-1]) for f in files]

    # Sort files using keys:
    files = [f for (k, f) in sorted(zip(keys, files))]

    return files 

def get_folders_sorted(path):
    '''
    Get all paths from a path with a star (using glob)
    '''
    from glob import glob
    # Get all folders if a '*' was used:
    paths = sorted(glob(path))
	
    if len(paths) == 0:
        raise Exception('No folders found at the specified path: ' + path)
	
    # Check if all paths are folders:	
    paths = [p for p in paths if os.path.isdir(p)]
    
    return paths
           
# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Utility functions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
           
def _sanity_check_(meta):
    '''
    Simple sanity check of the geometry record.
    '''
    minimum_set = ['det_pixel', 'src2det', 'src2obj', 'theta_max', 'theta_min', 'unit']

    for word in minimum_set:
        if (word not in meta['geometry']): raise ValueError('Missing ' + word + ' in the meta data. Something wrong with the log file?')
           
def _file_to_dictionary_(path, file_mask, separator = ':'):
    '''
    Read text file and return a dictionary with records.
    '''
    
    # Initialize records:
    records = {}
    
    #names = []
    
    # Try to find the log file in the selected path and file_mask
    log_file = [x for x in os.listdir(path) if (os.path.isfile(os.path.join(path, x)) and file_mask in os.path.join(path, x))]

    # Check if there is one file:
    if len(log_file) == 0:
        #warnings.warn('Log file not found in path: ' + path + ' *'+file_mask+'*')
        raise Exception('Log file not found @ ' + os.path.join(path, '*'+file_mask+'*'))    
        #return None
        
    if len(log_file) > 1:
        print('Found several log files. Currently using: ' + log_file[0])
        log_file = os.path.join(path, log_file[0])
    else:
        log_file = os.path.join(path, log_file[0])

    # Loop to read the file record by record:
    with open(log_file, 'r') as logfile:
        for line in logfile:
            name, var = line.partition(separator)[::2]
            name = name.strip().lower()
            
            # Dont mind empty lines and []:
            if re.search('[a-zA-Z]', name):
                if (name[0] != '['):
                    
                    # Remove \n:
                    var = var.rstrip()
                    
                    # If needed to separate the var and save the number of save the whole string:               
                    try:
                        var = float(var.split()[0])
                        
                    except:
                        var = var
                        
                    records[name] = var
                    
    if not records:
        raise Exception('Something went wrong during parsing the log file at:' + path)                    
        
    return records                

def _sample_image_(image, sample):
    '''
    Subsample the image or bin it if possible...
    '''
    
    if sample == 1:
        return image
    
    if sample % 2 != 0:
        warnings.warn('Sampling is not even. Won`t use binning.')
        image = image[::sample, ::sample]
        
    else:
        while sample > 1:
            
            if (image.dtype.kind == 'i') | (image.dtype.kind == 'u'):   
                image //= 4
            else:
                image /= 4
                
            image = (image[:-1:2, :] + image[1::2, :])
            image = (image[:, :-1:2] + image[:, 1::2]) 
            
            sample /= 2 
            
        return image

def _parse_unit_(string):
    '''
    Look for units in the string and return a factor that converts this unit to Si.
    '''
    
    # Here is what we are looking for:
    units_dictionary = {'nm':1e-6, 'nanometre':1e-6, 'um':1e-3, 'micrometre':1e-3, 'mm':1, 
                        'millimetre':1, 'cm':10.0, 'centimetre':10.0, 'm':1e3, 'metre':1e3, 
                        'rad':1, 'deg':numpy.pi / 180.0, 'ms':1, 's':1e3, 'second':1e3, 
                        'minute':60e3, 'us':0.001, 'kev':1, 'mev':1e3, 'ev':0.001,
                        'kv':1, 'mv':1e3, 'v':0.001, 'ua':1, 'ma':1e3, 'a':1e6, 'line':1}    
                        
    factor = [units_dictionary[key] for key in units_dictionary.keys() if key in string.split()]
    
    if factor == []: factor = 1
    else: factor = factor[0]

    return factor    

def _flexray_translate_(records):                  
    """
    Translate records parsed from the Flex-Ray log file (scan settings.txt) to the meta object.
    """
    # If the file was not found:
        
    if records is None: raise Exception('No records found!')
    
    # Initialize empty meta record:
    geom = init_geometry(geom_type=GEOM_STAOFF)    
    meta = init_meta(geom)
    
    # Dictionaries that describe the Flexray log record:        
    geom_dict =     {'img_pixel':'voxel size',
                     'det_pixel':'binned pixel size',
                    
                    'src2obj':'sod',
                    'src2det':'sdd',
                    
                    'src_vrt':'ver_tube',
                    'src_hrz':'tra_tube',
                    
                    'det_vrt':'ver_det',
                    'det_hrz':'tra_det',                    
                    
                    'axs_hrz':'tra_obj',
                    
                    'theta_max':'last angle',
                    'theta_min':'start angle',
                    
                    'roi':'roi (ltrb)'}
                    
    sett_dict =     {'voltage':'tube voltage',
                    'power':'tube power',
                    'averages':'number of averages',
                    'mode':'imaging mode',
                    'filter':'filter',
                    
                    'exposure':'exposure time (ms)',
                    
                    'binning':'binning value',
                    
                    'dark_avrg' : '# offset images',
                    'pre_flat':'# pre flat fields',
                    'post_flat':'# post flat fields'}
    
    descr_dict =    {'duration':'scan duration',
                    'name':'sample name',
                    'comments' : 'comment', 
                    
                    'samp_size':'sample size',
                    'owner':'sample owner',

                    'date':'date'}
    
    # Translate:
    geometry = meta['geometry']
    settings = meta['settings']
    description = meta['description']
       
    # Translate:
    _copydict_(geometry, records, geom_dict)
    _copydict_(settings, records, sett_dict)
    _copydict_(description, records, descr_dict)
    
    # binned pixel size can't be trusted in all logs... use voxel size:
    geometry['img_pixel'] *= _parse_unit_('um')
    geometry['det_pixel'] = numpy.round(geometry['img_pixel'] * (geometry['src2det'] / geometry['src2obj']), 5)    
    
    # Compute the center of the detector:
    roi = numpy.int32(geometry.get('roi').split(sep=','))
    geometry['roi'] = roi.tolist()

    _flex_motor_correct_(geometry, settings)
    
    # TODO: add support for piezo motors PI_X PI_Y
    
    return {'geometry':geometry, 'settings':settings, 'description':description}
        
def _flex_motor_correct_(geometry, settings):
    '''
    Apply some motor offsets to get to a correct coordinate system.
    '''
    # Correct some records (FlexRay specific):
    geometry['det2obj'] = geometry.get('src2det') - geometry.get('src2obj')
    
    # Horizontal offsets:
    geometry['det_hrz'] += 24    
    geometry['src_vrt'] -= 5

    # Rotation axis:
    geometry['axs_hrz'] -= 0.5
            
    # roi:        
    roi = geometry['roi']    
    centre = [(roi[0] + roi[2]) // 2 - 971, (roi[1] + roi[3]) // 2 - 767]
    
    # Take into account the ROI of the detector:
    #geometry['det_vrt'] -= centre[1] / settings.get('binning') * geometry['det_pixel']
    #geometry['det_hrz'] -= centre[0] / settings.get('binning') * geometry['det_pixel']
    # Not sure the binning should be taken into account...
    geometry['det_vrt'] -= centre[1] * geometry['det_pixel']
    geometry['det_hrz'] -= centre[0] * geometry['det_pixel']
    
    geometry['vol_tra'][0] = (geometry['det_vrt'] * geometry['src2obj'] + geometry['src_vrt'] * geometry['det2obj']) / geometry.get('src2det')

def _modify_astra_vector_(proj_geom, geometry):
    """
    Modify ASTRA vector using known offsets from the geometry records.
    """
    # Even if the geometry is of the type 'simple' (GEOM_SYMPLE), we need to generate ASTRA vector to be able to rotate the reconstruction volume if needed.
    thetas = proj_geom['ProjectionAngles']
    theta_count = len(thetas)
    
    det_pixel = geometry['det_pixel'] * numpy.array(geometry.get('proj_sample'))[::2]
    
    # Compute current offsets (for this angle):
    if geometry.get('type') == GEOM_SIMPLE:
        
        src2obj = geometry.get('src2obj')
        det2obj = geometry.get('det2obj')
        det_vrt = 0 
        det_hrz = 0
        det_mag = 0
        det_yaw = 0
        det_roll = 0
        det_pitch = 0
        src_vrt = 0
        src_hrz = 0
        src_mag = 0
        axs_hrz = 0
        axs_mag = 0
        axs_rot = [0.0, 0.0]
    
    # Compute current offsets:
    elif geometry.get('type') == GEOM_STAOFF:
        
        src2obj = geometry['src2obj']
        det2obj = geometry['det2obj']
        det_vrt = geometry['det_vrt'] 
        det_hrz = geometry['det_hrz'] 
        det_mag = geometry['det_mag'] 
        det_roll = geometry['det_roll'] 
        det_yaw = geometry['det_yaw']
        det_pitch = geometry['det_pitch']
        src_vrt = geometry['src_vrt'] 
        src_hrz = geometry['src_hrz'] 
        src_mag = geometry['src_mag'] 
        axs_hrz = geometry['axs_hrz'] 
        axs_mag = geometry['axs_mag'] 
        axs_rot = geometry['axs_rot'] 
      
    # Use linear offsets:    
    elif geometry.get('type') == GEOM_LINOFF:
        b = numpy.linspace(0, 1, theta_count)
        a = 1 - b
        src2obj = geometry['src2obj']
        det2obj = geometry['det2obj']
        det_vrt = geometry['det_vrt'][0] * a + geometry['det_vrt'][1] * b
        det_hrz = geometry['det_hrz'][0] * a + geometry['det_hrz'][1] * b  
        det_mag = geometry['det_mag'][0] * a + geometry['det_mag'][1] * b  
        det_roll = geometry['det_roll'][0] * a + geometry['det_roll'][1] * b  
        det_yaw = geometry['det_yaw'][0] * a + geometry['det_yaw'][1] * b  
        det_pitch = geometry['det_pitch'][0] * a + geometry['det_pitch'][1] * b  
        src_vrt = geometry['src_vrt'][0] * a + geometry['src_vrt'][1] * b 
        src_hrz = geometry['src_hrz'][0] * a + geometry['src_hrz'][1] * b 
        src_mag = geometry['src_mag'][0] * a + geometry['src_mag'][1] * b 
        axs_hrz = geometry['axs_hrz'][0] * a + geometry['axs_hrz'][1] * b 
        axs_mag = geometry['axs_mag'][0] * a + geometry['axs_mag'][1] * b 
        axs_rot = geometry['axs_rot'][0] * a + geometry['axs_rot'][1] * b 
        
    else: raise ValueError('Wrong geometry type: ' + geometry.get('type'))
    
    # We are going to generate vectors instead of taking them from ASTRA
    axis = numpy.dot(transforms3d.euler.euler2mat(axs_rot[0], axs_rot[1], 0), [0,0,1])
    
    src_vect0 = numpy.array([0, -src2obj, 0]) 
    det_vect0 = numpy.array([0, det2obj, 0])
    
    det_axis_hrz0 = numpy.array([det_pixel[1], 0, 0])
    det_axis_vrt0 = numpy.array([0, 0, det_pixel[0]])
    
    src_vect = numpy.zeros([theta_count, 3])
    det_vect = numpy.zeros([theta_count, 3])
    det_axis_hrz = numpy.zeros([theta_count, 3])
    det_axis_vrt = numpy.zeros([theta_count, 3])
    
    proj_geom = astra.geom_2vec(proj_geom)
    
    # Genertate initial circular orbit:
    for ii, theta in enumerate(thetas):
        
        # Rotation matrix:
        Ri = transforms3d.euler.axangle2mat(axis, theta)
        
        src_vect = numpy.dot(Ri, src_vect0)
        det_vect = numpy.dot(Ri, det_vect0)
        
        det_axis_hrz = numpy.dot(Ri, det_axis_hrz0)
        det_axis_vrt = numpy.dot(Ri, det_axis_vrt0)
    
        # Define vectors:
        #src_vect = vectors[ii, 0:3]    
        #det_vect = vectors[ii, 3:6]    
        #det_axis_hrz = vectors[ii, 6:9]          
        #det_axis_vrt = vectors[ii, 9:12]
            
        #Precalculate vector perpendicular to the detector plane:
        det_normal = numpy.cross(det_axis_hrz, det_axis_vrt)
        det_normal = det_normal / numpy.sqrt(numpy.dot(det_normal, det_normal))
        
        # Translations relative to the detecotor plane:
    
        #Detector shift (V):
        det_vect += det_vrt * det_axis_vrt / det_pixel[0]

        #Detector shift (H):
        det_vect += det_hrz * det_axis_hrz / det_pixel[1]

        #Detector shift (M):
        det_vect += det_mag * det_normal /  det_pixel[1]

        #Source shift (V):
        src_vect += src_vrt * det_axis_vrt / det_pixel[0]

        #Source shift (H):
        src_vect += src_hrz * det_axis_hrz / det_pixel[1]

        #Source shift (M):
        src_vect += src_mag * det_normal / det_pixel[1] 

        # Rotation axis shift:
        det_vect -= axs_hrz * det_axis_hrz  / det_pixel[1]
        src_vect -= axs_hrz * det_axis_hrz  / det_pixel[1]
        det_vect -= axs_mag * det_normal /  det_pixel[1]
        src_vect -= axs_mag * det_normal /  det_pixel[1]

        # Rotation relative to the detector plane:
        # Compute rotation matrix

        # Detector roll:    
        T = transforms3d.axangles.axangle2mat(det_normal, det_roll)
        det_axis_hrz[:] = numpy.dot(T, det_axis_hrz)
        det_axis_vrt[:] = numpy.dot(T, det_axis_vrt)
        
        # Detector yaw:    
        T = transforms3d.axangles.axangle2mat(det_axis_vrt, det_yaw)
        det_axis_hrz[:] = numpy.dot(T, det_axis_hrz)
        
        # Detector pitch:    
        T = transforms3d.axangles.axangle2mat(det_axis_hrz, det_pitch)
        det_axis_vrt[:] = numpy.dot(T, det_axis_vrt)
    
        # Global transformation:
        # Rotation matrix based on Euler angles:
        R = transforms3d.euler.euler2mat(geometry['vol_rot'][0], geometry['vol_rot'][1], geometry['vol_rot'][2], 'rzyx')

        # Apply transformation:
        det_axis_hrz = numpy.dot(det_axis_hrz, R)
        det_axis_vrt = numpy.dot(det_axis_vrt, R)
        src_vect = numpy.dot(src_vect,R)
        det_vect = numpy.dot(det_vect,R)            
                
        # Add translation:
        vect_norm = numpy.sqrt((det_axis_vrt ** 2).sum())

        # Take into account that the center of rotation should be in the center of reconstruction volume:        
        T = numpy.array([geometry['vol_tra'][1] * vect_norm / det_pixel[1], geometry['vol_tra'][2] * vect_norm / det_pixel[1], geometry['vol_tra'][0] * vect_norm / det_pixel[0]])    
        
        src_vect -= numpy.dot(T, R)           
        det_vect -= numpy.dot(T, R)
        
        # Revrite vectors:
        proj_geom['Vectors'][ii, 0:3] = src_vect
        proj_geom['Vectors'][ii, 3:6] = det_vect
        proj_geom['Vectors'][ii, 6:9] = det_axis_hrz
        proj_geom['Vectors'][ii, 9:12] = det_axis_vrt
                    
    return proj_geom

def _metadata_translate_(records):                  
    """
    Translate records parsed from the Flex-Ray log file (scan settings.txt) to the meta object
    """
    # If the file was not found:
    if records is None: raise Exception('No records found!')
    
    # Initialize empty meta record:
    geom = init_geometry(geom_type=GEOM_STAOFF)    
    meta = init_meta(geom)
        
    # Dictionaries that describe the metadata.toml record:        
    geom_dict = {'det_pixel':'detector pixel size',
                
                'src2obj':'sod',
                'src2det':'sdd',
                
                'src_vrt':'ver_tube',
                'src_hrz':'tra_tube',
                
                'det_vrt':'ver_det',
                'det_hrz':'tra_det',                    
                
                'axs_hrz':'tra_obj',
                
                'theta_max':'last_angle',
                'theta_min':'first_angle',
                
                'roi':'roi'}
    
    sett_dict = {'voltage':'kv',
                    'power':'power',
                    'focus':'focusmode',
                    'averages':'averages',
                    'mode':'mode',
                    'filter':'filter',
                    
                    'exposure':'exposure',
                    
                    'dark_avrg' : 'dark',
                    'pre_flat':'pre_flat',
                    'post_flat':'post_flat'}

    descr_dict = {'duration':'total_scantime',
                    'name':'scan_name'} 
        
    geometry = meta['geometry']
    settings = meta['settings']
    description = meta['description']
    
    # Copy:
    _copydict_(geometry, records, geom_dict)
    _copydict_(settings, records, sett_dict)
    _copydict_(geometry, records, descr_dict)
            
    # Compute the center of the detector:
    roi = re.sub('[] []', '', geometry['roi']).split(sep=',')
    roi = numpy.int32(roi)
    geometry['roi'] = roi.tolist()
    
    # Detector pixel is not changed here when binning mode is on...
    if (settings['mode'] == 'HW2SW1High')|(settings['mode'] == 'HW1SW2High'):
        geometry['det_pixel'] *= 2    
    elif (settings['mode'] == 'HW2SW2High'):
        geometry['det_pixel'] *= 4   

    geometry['img_pixel'] = geometry['det_pixel'] / (geometry['src2det'] / geometry['src2obj'])    

    _flex_motor_correct_(geometry, settings)
        
    # Populate meta:    
    meta = {'geometry':geometry, 'settings':settings, 'description':description}
        
    return meta

def _copydict_(destination, source, dictionary):
    """
    Copy dictionary to dictionary
    """
    for key in dictionary.keys():
        
        if dictionary[key] in source.keys():
            #destination[key] = source[dictionary[key]]
            
            # Remove spaces:
            val = source[dictionary[key]]
            if type(val) is str:
                val = val.strip()
                
            destination[key] = val
            
        else:
            warnings.warn('Record is not found: ' + dictionary[key])
            
def _check_success_(proj, meta, success):
    """
    If few files are missing - interpolate, if many - adjust theta record in meta
    """
    success = numpy.array(success)
    
    if len(success) == sum(success):
        return proj
    
    # Check if failed projections come in bunches or singles:
    fails = numpy.where(success == 0)[0]
    
    if fails.size == 1:
        print('One projection is missing, we will try to interpoolate.')
        ii = fails[0]
        proj[ii] = (proj[ii - 1] + proj[ii + 1]) / 2
        
    else:
        if min(fails[1:] - fails[:-1]) > 1:
            print('Few projections are missing, we will try to interpoolate them.')
            
            # Very simple interpolation:
            for ii in fails:
                proj[ii] = (proj[ii - 1] + proj[ii + 1]) / 2
            
        else:
            print('Some clusters of projections are missing. We will adjust the thetas record.')
            
            thetas = numpy.linspace(meta['geometry']['theta_min'], meta['geometry']['theta_max'], len(success))
            
            meta['geometry']['_thetas_'] = thetas[success == 1]
            proj = proj[success == 1]
            
    return proj
