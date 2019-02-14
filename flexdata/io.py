#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kostenko
Created on Oct 2018

This module contains read / write routines for stacks of images and some parsers for log file reading.

Most of the basic image formats are supported through imageio module + raw and matlab binary files.

"""

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Imports >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

import numpy          # arrays arrays arrays
import os             # operations with filenames
import re             # findall function
import warnings       # warn me if we are in trouble!
import imageio        # io for images
import psutil         # RAM tester
import toml           # TOML format parcer
from tqdm import tqdm # Progress barring
import time           # Pausing

from scipy.io import loadmat # Reading matlab format
from . import array          # operations with arrays
from . import geometry       # geometry classes

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     
def stack_shape(path, name, skip = 1, sample = 1, shape = None, dtype = None, format = None):
    """
    Determine the shape of stack on disk.
    
    Args:
        path (str): path to the files location
        name (str): common part of the files name
        skip (int): read every so many files
        sample (int): sampling factor in x/y direction
        shape (array): shape of the files. Use it when the format is 'raw'.
        dtype (str or numpy.dtype): data type of the files
        format (str): file format ('tif', 'raw', etc)
    """  
    # Retrieve file names, sorted by name
    files = get_files_sorted(path, name)    
    
    if len(files) == 0: raise IOError('Files not found at:', os.path.join(path, name))
    
    # Apply skip:
    files = files[::skip]
    
    # Confirm the shape and the dtype of the image:
    image = read_image(files[0], sample, shape, format, dtype)
    
    # Create a mapped array if needed:
    #if shape and sample are used, shape means the shape of the image on disk before subsampling
    shape = (len(files), image.shape[0], image.shape[1])
    dtype = image.dtype
    
    return shape, dtype
    
def read_stack(path, name, skip = 1, sample = 1, shape = None, dtype = None, format = None, flipdim = False, memmap = None, success = None):    
    """
    Read stack of files and return a numpy array.
    
    Args:
        path (str): path to the files location
        name (str): common part of the files name
        skip (int): read every so many files
        sample (int): sampling factor in x/y direction
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
    files = files[::skip]
    file_n = len(files)
                    
    # Create a mapped array if needed:
    #if shape and sample are used, shape means the shape of the image on disk before subsampling
    shape_samp, dtype = stack_shape(path, name, skip, sample, shape, dtype, format)
    
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
        if not success is None:
            try:
                im = read_image(files[k], sample, shape, format, dtype)
                data[k, :, :] = im     
                success[k] = 1
                
            except Exception:
                success[k] = 0            
                    
                warnings.warn('Error reading file:' + files[k])
                pass
        else:
            im = read_image(files[k], sample, shape, format, dtype)    

            data[k, :, :] = im
 
    time.sleep(0.3) # This is needed to let print message be printed before the next porogress bar is created
    
    # Get rid of the corrupted data:
    if not success is None:                
        if sum(success) != file_n:
            warnings.warn('%u files are CORRUPTED!'%(file_n - sum(success)))
                    
        print(f'%u files were loaded. %u%% memory left (%u GB).' % (sum(success), free_memory(True), free_memory(False)))
    else:
        print(f'%u files were loaded. %u%% memory left (%u GB).' % (len(files), free_memory(True), free_memory(False)))
    
    # Apply dimension switch:    
    if flipdim:    
        data = array.raw2astra(data)    
    
    return data
                 
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
    time.sleep(0.3)                            
    
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
            
def read_flexray(path, sample = 1, skip = 1, memmap = None, proj_number = None):
    '''
    Read projecition data for the FLex-Ray scaner. Read, dark-, flat-field images and scan parameters.
    
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
        geom (geometry)   : description of the geometry, physical settings and comments
        
    '''
    
    dark = read_stack(path, 'di00', skip, sample, dtype = 'float32')
    flat = read_stack(path, 'io00', skip, sample, dtype = 'float32')
    
    # Read the raw data. Use success array to check for corrupted files.
    if proj_number:
        success = numpy.zeros(int(numpy.ceil(proj_number / skip)))        
    else:
        success = None
        
    proj = read_stack(path, 'scan_', skip, sample, dtype = 'float32', memmap = memmap, success = success)
    
    # Try to retrieve metadata:
    try:
        geom = read_flexraymeta(path, sample)   

    except:
        
        geom = read_flexraylog(path, sample)  
        
    # Check success. If a few files were not read - interpolate, otherwise adjust the meta record.
    proj = _check_success_(proj, geom, success)
    
    return proj, flat, dark, geom            
    
def write_image(filename, image, compress = 0):
    """
    Write a single image. Use compression if needed (0-9).
    """     
    with imageio.get_writer(filename) as w:
        w.append_data(image, {'compress': compress})

def read_image(file, sample = 1, shape = None, format = None, dtype = None):
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
            raise Exception("Can't find extension of the file: " + file + '\n Use format to provide file format.')
            
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
         
    im = _sample_image_(im, sample)
    return im    
    
def read_flexraylog(path, sample = 1):
    """
    Read the log file of FLexRay scanner and return dictionaries with parameters of the scan.
    
    Args:
        path   (str): path to the files location
        sample (int): subsampling of the input data
        
    Returns:    
        geometry    : circular geometry class
    """   
    # Dictionary that describes the Flexray log record:        
    param_dict =     {'img_pixel':'voxel size',
                     
                    'src2obj':'sod',
                    'src2det':'sdd',
                    
                    'src_ort':'ver_tube',
                    'src_tan':'tra_tube',
                    
                    'det_ort':'ver_det',
                    'det_tan':'tra_det',                    
                    
                    'axs_tan':'tra_obj',
                    
                    'theta_max':'last angle',
                    'theta_min':'start angle',
                    
                    'roi':'roi (ltrb)',
                    
                    'voltage':'tube voltage',
                    'power':'tube power',
                    'averages':'number of averages',
                    'mode':'imaging mode',
                    'filter':'filter',
                    
                    'exposure':'exposure time (ms)',
                    
                    'binning':'binning value',
                    
                    'dark_avrg' : '# offset images',
                    'pre_flat':'# pre flat fields',
                    'post_flat':'# post flat fields',
    
                    'duration':'scan duration',
                    'name':'sample name',
                    'comments' : 'comment', 
                    
                    'samp_size':'sample size',
                    'owner':'sample owner',

                    'date':'date'}
    
    # Read file and translate:
    records = file_to_dictionary(os.path.join(path, 'scan settings.txt'), separator = ':', translation = param_dict)
    
    # Corrections specific to this type of file:
    records['img_pixel'] *= _parse_unit_('um')
    
    roi = numpy.int32(records.get('roi').split(sep=',')).tolist()
    records['roi'] = roi
    
    # Initialize geometry:
    geom = geometry.circular()
    geom.from_dictionary(records)
    
    geom.parameters['det_pixel'] *= sample
    geom.parameters['img_pixel'] *= sample
    
    # Some Flexray scanner-specific motor offset corrections:
    _flex_motor_correct_(geom)
        
    return geom

def read_flexraymeta(path, sample = 1):
    """
    Read the metafile produced by Flexray scripting.
    
    Args:
        path   (str): path to the files location
        sample (int): subsampling of the input data
        
    Returns:    
        geometry    : circular geometry class
    """      
    param_dict = {'det_pixel':'detector pixel size',
                
                'src2obj':'sod',
                'src2det':'sdd',
                
                'src_ort':'ver_tube',
                'src_tan':'tra_tube',
                
                'det_ort':'ver_det',
                'det_tan':'tra_det',                    
                
                'axs_tan':'tra_obj',
                
                'theta_max':'last_angle',
                'theta_min':'first_angle',
                
                'roi':'roi',
    
                'voltage':'kv',
                'power':'power',
                'focus':'focusmode',
                'averages':'averages',
                'mode':'mode',
                'filter':'filter',
                
                'exposure':'exposure',
                
                'dark_avrg' : 'dark',
                'pre_flat':'pre_flat',
                'post_flat':'post_flat',

                'duration':'total_scantime',
                'name':'scan_name'} 
    
    records = file_to_dictionary(os.path.join(path, 'metadata.toml'), separator = '=', translate = param_dict)        
    
    # Compute the center of the detector:
    roi = re.sub('[] []', '', records['roi']).split(sep=',')
    roi = numpy.int32(roi)
    records['roi'] = roi.tolist()
    
    # Detector pixel is not changed here when binning mode is on...
    if (records['mode'] == 'HW2SW1High')|(records['mode'] == 'HW1SW2High'):
        records['det_pixel'] *= 2    
        records['img_pixel'] *= 2    
        
    elif (records['mode'] == 'HW2SW2High'):
        records['det_pixel'] *= 4   
        records['img_pixel'] *= 4    
    
    # Initialize geometry:
    geom = geometry.circular()
    geom.from_dictionary(records)
    
    geom.parameters['det_pixel'] *= sample
    geom.parameters['img_pixel'] *= sample
    
    return geom
    
def read_geometry(path, sample = 1):
    '''
    Read a native meta file.
    
    Args:
        path   (str): path to the file location
        sample (int): subsampling of the input data
        
    Returns:    
        geometry    : circular geometry class
    '''    
    records = read_toml(os.path.join(path, 'geometry.toml'))   
    records['det_pixel'] *= sample
    records['img_pixel'] *= sample
              
    # Initialize geometry:
    geom = geometry.circular()
    geom.from_dictionary(records)    
    
    return geom

def file_to_dictionary(file_path, separator = ':', translation = None):
    '''
    Read a text file and return a dictionary with records.
    
    Args:
        file_path (str): file to read
        separator (str): separator between the keys and values
        translation (dict): dictionary for translating initial keys to a new naming
    '''
    
    # Initialize records:
    records = {}
        
    # Check if there is one file:
    if not os.path.isfile(file_path):
        raise Exception('Log file not found @ ' + file_path)    
    
    # Loop to read the file record by record:
    with open(file_path, 'r') as logfile:
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
                        
                    # Remove spaces:
                    if type(var) is str:
                        var = var.strip()    
                        
                    records[name] = var
                    
    if not records:
        raise Exception('Something went wrong during parsing the log file at:' + file_path)                    
       
    # Translate from one dictionaty to another: 
    if translation:
        
        records_t = {}
        
        for key in translation.keys():
            records_t[key] = records.get(translation[key])
            
        return records_t
    else:
        return records
       
def read_toml(file_path):
    """
    Read a toml file.
    
    Args:
        file_path (str): read file form that location
    """  
    # TOML is terrible sometimes... which is why we are doing some fixing of the file that is loaded before parsing
    file = open(file_path,'r') 
    
    s = file.read()
    file.close()
    
    s = s.replace(u'\ufeff', '')
    s = s.replace('(', '[')
    s = s.replace(')', ']')
        
    record = toml.loads(s)
    
    # Somehow TOML doesnt support numpy. Here is a workaround:
    for key in record.keys():
        if isinstance(record[key], dict):
            for subkey in record[key].keys():
                record[key][subkey] = _python2numpy_(record[key][subkey])
        else:
            record[key] = _python2numpy_(record[key])

    return record        
        
def write_toml(filename, record):
    """
    Write a toml file.
    
    Args:
        filename (str): location to write the file to
        record (dict, geometry): geomety class record or an arbitrary dictionary    
    """
    # Convert to dictionary:
    if not(type(record) is dict):
        record = record.to_dictionary()
        
    # Make path if does not exist:
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)

    # It looks like TOML module doesnt like numpy arrays and numpy types. 
    # Use lists and native types for TOML.
    #for key in record.keys():
    #    if isinstance(record[key], dict):
    #        for subkey in record[key].keys():
    #            record[key][subkey] = _numpy2python_(record[key][subkey])
    #    else:
    #        record[key] = _numpy2python_(record[key])
            
    # Save TOML to a file:
    with open(filename, 'w') as f:
        d = toml.dumps(record)
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
        
def write_astra(filename, data_shape, geom):
    """
    Write an astra-readable projection geometry vector.
    """    
    # Make path if does not exist:
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)
    
    numpy.savetxt(filename, geom.astra_projection_geom(data_shape)['Vectors']) 
        
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
    
def _flex_motor_correct_(geom):
    '''
    Apply some motor offsets to get to a correct coordinate system.
    '''
    # Correct some records (FlexRay specific):
    
    # Horizontal offsets:
    geom.parameters['det_tan'] += 24    
    geom.parameters['src_ort'] -= 5

    # Rotation axis:
    geom.parameters['axs_tan'] -= 0.5
            
    # roi:        
    roi = geom.description['roi']
    centre = [(roi[0] + roi[2]) // 2 - 971, (roi[1] + roi[3]) // 2 - 767]
    
    # Not sure the binning should be taken into account...
    geom.parameters['det_ort'] -= centre[1] * geom.parameters['det_pixel']
    geom.parameters['det_tan'] -= centre[0] * geom.parameters['det_pixel']
    
    geom.parameters['vol_tra'][0] = (geom.parameters['det_ort'] * geom.src2obj + 
                   geom.parameters['src_ort'] * geom.det2obj) / geom.src2det
           
def _check_success_(proj, geom, success):
    """
    If few files are missing - interpolate, if many - adjust theta record in meta
    """
    if success is None: 
        return proj
    
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
            
            thetas = numpy.linspace(geom.range[0], geom.range[1], len(success))
            geom.parameters['_thetas_'] = thetas[success == 1]
            proj = proj[success == 1]
            
    return proj
