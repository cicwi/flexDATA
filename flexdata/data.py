#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains some input / output routines for stacks of images and parsers for translating scanner log files into geometry definitions.

Most of the basic image formats are supported through imageio module.
Raw binaries and matlab binary files can be loaded.

Utility functions to hande big arrays of data. All routines support memmap arrays.
However, some operations will need enough memory for at least one copy of the data for intermediate
results. This can be improved through better use of memmaps.

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
import logging
from scipy.io import loadmat # Reading matlab format
from . import geometry       # geometry classes
from .correct import correct_roi
# >>>>>>>>>>>>>>>>>>>> LOGGER CLASS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


class logger:
   """
   A class for logging and printing messages.
   """
   # Save messages to a log file:
   file = ''

   @staticmethod
   def _write_(message):
       """
       Dump message into a file if it is available.
       """
       # Add timestamp:
       message = '[%s]: %s' % (time.asctime(), message)

       # Write:
       if logger.file:
           with open(logger.file, 'w') as file:
               file.write(message)

   @staticmethod
   def print(message):
      """
      Simply prints and saves a message.
      """
      print(message)

   @staticmethod
   def title(message):
      """
      Print something important.
      """
      print('')
      print(message)
      print('')

   @staticmethod
   def warning(message):
      """
      Raise a warning.
      """
      warnings.warn(message)

   @staticmethod
   def error(message):
      """
      Raise an error.
      """
      raise Exception(message)


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

def read_stack(path, name, skip = 1, sample = 1, shape = None, dtype = None,
               format = None, transpose = [1, 0, 2], updown = True, memmap = None, success = None):
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
        data = numpy.memmap(memmap, dtype=dtype, mode='w+', shape = shape_samp)

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
    data = flipdim(data, transpose, updown)

    return data

def write_stack(path, name, data, dim = 1, skip = 1, dtype = None, zip = False, format = 'tiff', updown = False):
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
        sl = anyslice(data, ii * skip, dim)
        img = data[sl]

        if updown:
            img = img[::-1, :]

        # Cast data to another type if needed
        if dtype is not None:
            img = cast2type(img, dtype, bounds)

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

def read_flexraylog(path, *args):
   warnings.warn("""
read_flexraylog is depecrated.

This function combined too much functionality. If you want similar functionality to what
read_flexraylog provided, use:
>>> from flexdata import data
>>> from flexdata import correct
>>> geom = data.parse_flexraylog(path, sample=binning)
>>> geom = correct.correct(geom,
                           profile='cwi-flexray-2019-04-24',
                           do_print_changes=True)
>>> geom = correct.correct_vol_center(geom)


""", DeprecationWarning, stacklevel=2)
   raise NotImplementedError()


def parse_flexraylog(path, sample = 1):
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

    geom = correct_roi(geom)

    if sample != 1:
        msg = f"Adjusted geometry by binning by {sample}"
        logging.info(msg)
        geom.log(msg)

    return geom


def read_flexraymeta(*args):
   warnings.warn("""
read_flexraymeta is depecrated.

This function combined too much functionality. If you want similar functionality to what
read_flexraymeta provided, use:
>>> from flexdata import data
>>> from flexdata import correct
>>> geom = data.parse_flexraymeta(path, sample=binning)
>>> geom = correct.correct(geom,
                           profile='cwi-flexray-2019-04-24',
                           do_print_changes=True)
>>> geom = correct.correct_vol_center(geom)


""", DeprecationWarning, stacklevel=2)
   raise NotImplementedError()


def parse_flexraymeta(path, sample = 1):
    """
    Read the metafile produced by the Flexray script generator.

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

    records = file_to_dictionary(os.path.join(path, 'metadata.toml'), separator = '=', translation = param_dict)

    # Compute the center of the detector:
    roi = re.sub('[] []', '', records['roi']).split(sep=',')
    roi = numpy.int32(roi)
    records['roi'] = roi.tolist()

    # Detector pixel is not changed here when binning mode is on...
    pixel_adjustment = 1
    if (records['mode'] == 'HW2SW1High')|(records['mode'] == 'HW1SW2High'):
        records['det_pixel'] *= 2
        records['img_pixel'] *= 2
        pixel_adjustment = 2

    elif (records['mode'] == 'HW2SW2High'):
        records['det_pixel'] *= 4
        records['img_pixel'] *= 4
        pixel_adjustment = 4

    # Check version
    version = records.get('FLEXTOML_VERSION')
    if version is None:
       logger.warning(f"No version for toml file. Expected {geom.FLEXTOML_VERSION}")
    elif version != geometry.FLEXTOML_VERSION:
       logger.warning(f"Version {version} found for toml file. Expected: {geom.FLEXTOML_VERSION}")

    # Initialize geometry:
    geom = geometry.circular()
    geom.from_dictionary(records)

    if pixel_adjustment != 1:
       msg = f"Adjusted pixel size by {pixel_adjustement} due to {records['mode']}"
       logging.info(msg)
       geom.log(msg)

    geom.parameters['det_pixel'] *= sample
    geom.parameters['img_pixel'] *= sample

    if sample != 1:
        msg = f"Adjusted geometry by binning by {sample}"
        logging.info(msg)
        geom.log(msg)

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

    # Check version
    version = records.get('FLEXTOML_VERSION')
    if version is None:
       logger.warning(f"No version for toml file. Expected {geometry.FLEXTOML_VERSION}")
    elif version != geometry.FLEXTOML_VERSION:
       logger.warning(f"Version {version} found for toml file. Expected: {geometry.FLEXTOML_VERSION}")

    # Initialize geometry:
    geom = geometry.circular()
    geom.from_dictionary(records)

    return geom

def geom_diff(geom1, geom2, full_diff=False):
    '''
    Returns a dictionary with changed values in two geometries. Diff is computed
    per item as: geom1[item] - geom2[item].

    If `full_diff` is `True`, also returns added, removed and unchanged dict items.

    Useful if you'd like to see if corrections have been made in one geometry
    with respect to a second another.

    Either provide a path, geometry object or dictionary to `geom2` or `geom1`.

    :param geom1: First path, geometry object or dictionary.
    :param geom2: Second path, geometry object or dictionary.
    :param full_diff: If set to `True` also shows added, removed and unchanged.
    :return: Geometry dictionary
    '''

    from flexdata.geometry import basic

    def input_to_dict(input, var_name):
        if isinstance(input, str):
            input = read_toml(input)
        elif isinstance(input, basic):
            input = input.to_dictionary()
        elif isinstance(input, dict):
            pass
        else:
            raise ValueError("`" + var_name + "` must be a path, dict or a geometry type.")

        return input

    geom1 = input_to_dict(geom1, 'geom1')
    geom2 = input_to_dict(geom2, 'geom2')

    # Source: https://stackoverflow.com/questions/1165352
    class DictDiffer(object):
        def __init__(self, current_dict, past_dict):
            self.current_dict, self.past_dict = current_dict, past_dict
            self.set_current, self.set_past = set(current_dict.keys()), set(past_dict.keys())
            self.intersect = self.set_current.intersection(self.set_past)

        def added(self):
            return self.set_current - self.intersect

        def removed(self):
            return self.set_past - self.intersect

        def changed(self):
            return set(o for o in self.intersect if self.past_dict[o] != self.current_dict[o])

        def diff(self):
            diff = {}

            for key in self.changed():
                diff[key] = numpy.subtract(self.current_dict[key], self.past_dict[key])

            return diff

        def unchanged(self):
            return set(o for o in self.intersect if self.past_dict[o] == self.current_dict[o])

    dd = DictDiffer(geom2, geom1)

    if full_diff:
        return {
            'added': dd.added(),
            'removed': dd.removed(),
            'changed': dd.changed(),
            'diff_changed': dd.diff(),
            'unchanged': dd.unchanged(),
        }
    else:
        return dd.diff()

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
    Return amount of free RAM memory in GB.

    Args:
        percent (bool): percentage of the total or in GB.

    '''
    if not percent:
        return psutil.virtual_memory().available/1e9

    else:
        return psutil.virtual_memory().available / psutil.virtual_memory().total * 100

def free_disk(path):
    '''
    Return amount of free memory on disk in GB.

    Args:
        percent (bool): percentage of the total or in GB.

    '''
    statvfs = os.statvfs(path)
    return statvfs.f_frsize * statvfs.f_bavail/1e9

def gradient(array, axes = [0,1,2]):
    '''
    Compute the gradient of an array.

    Args:
        axes   : list of axes to apply gradient to.

    Returns:
        ndarray: shape = (3, k, l, m) where k,l,m - dimensions of the input array.
    '''
    num_dims = len(array.shape)
    d = []
    for ax in axes:
        pad_pattern = [(0, 0)] * num_dims
        pad_pattern[ax] = (0, 1)
        temp_d = numpy.pad(array, pad_pattern, mode='edge')
        temp_d = numpy.diff(temp_d, n=1, axis=ax)
        d.append(temp_d)

    return numpy.stack(d)

def divergence(array, axes = [0, 1, 2]):
    '''
    Compute the divergence of an array.

    Args:
        axes   : list of axes where the divergence is applied.

    Returns:
        ndarray: divergence of the input array
    '''

    num_dims = len(array.shape)-1
    for ii, ax in enumerate(axes):
        pad_pattern = [(0, 0)] * num_dims
        pad_pattern[ax] = (1, 0)
        temp_d = numpy.pad(array[ax, ...], pad_pattern, mode='edge')
        temp_d = numpy.diff(temp_d, n=1, axis=ax)
        if ii == 0:
            final_d = temp_d
        else:
            final_d += temp_d

    return final_d

def convolve_filter(array, filtr):
    """
    Apply a filter defined in Fourier space (a CTF) via convolution.

    Args:
        array : data array (implicit)
        filtr : a filter defined in Fourier space (1D - 3D)
    """
    if filtr.ndim == 3:
        axes = (0, 1, 2)
        x = numpy.fft.fftn(array, axes = axes) * filtr

    elif filtr.ndim == 2:
        axes = (0, 2)
        x = numpy.fft.fftn(array, axes = axes) * filtr[:, None, :]
    else:
        axes = (2,)
        x = numpy.fft.fftn(array, axes = axes) * filtr[None, None, :]

    x = numpy.abs(numpy.fft.ifftn( x , axes = axes))
    array[:] = x

def convolve_kernel(array, kernel):
    """
    Apply a kernel defined in real space (center in the center of the array) via convolution.

    Args:
        array (ndarray)  : data array (implicit)
        kernel(ndarray)  : real space kernel (1D - 3D)
    """
    if kernel.ndim == 3:
        axes = (0, 1, 2)

    elif kernel.ndim == 2:
        axes = (0, 2)

    else:
        axes = (2,)

    kernel = numpy.fft.fftshift(kernel, axes = axes)
    kernel = numpy.fft.fftn(kernel, axes = axes).conj()
    convolve_filter(array, kernel)

def autocorrelation(array, axes = (0,1,2)):
    '''
    Compute autocorrelation.
    '''
    x = numpy.fft.fftn(array, axes = axes)
    x *= x.conj()
    x = numpy.real(numpy.fft.ifftn(x, axes = axes))
    x = numpy.fft.fftshift(x, axes = axes)
    array[:] = x

def deconvolve_filter(array, filtr, epsilon):
    """
    Inverse convolution with Tikhonov regularization.

    Args:
        array (ndarray)  : data array (implicit)
        filtr (ndarray)  : Fourier space filter (1D - 3D)
        epsilon          : regularization parameter
        axes             : list of axes to apply deconvolution to.
    """
    if filtr.ndim == 3:
        axes = (0, 1, 2)

    elif filtr.ndim == 2:
        axes = (0, 2)
        filtr = filtr[:, None, :]
    else:
        axes = (2,)
        filtr = filtr[None, None, :]

    x = numpy.fft.fftn(array, axes = axes) * filtr.conj() / (numpy.abs(filtr) ** 2 + epsilon)
    x = numpy.abs(numpy.fft.ifftn( x , axes = axes))
    array[:] = x

def deconvolve_kernel(array, kernel, epsilon, axes = (0, 2)):
    """
    Inverse convolution with Tikhonov regularization.

    Args:
        array (ndarray)  : data array (implicit)
        filtr (ndarray)  : Fourier space filter (1D - 3D)
        epsilon          : regularization parameter
        axes             : list of axes to apply deconvolution to.
    """
    kernel = numpy.fft.fftshift(kernel, axes)
    kernel = numpy.fft.fftn(kernel, axes).conj()
    deconvolve_filter(array, kernel, epsilon, axes)

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

    Args:
        array: input array
        dim  : dim to apply the ramp
        width: width of the ramp
        mode :'linear' - creates linear decay of intensity; 'edge' - smears data in a costant manner; 'zero' - sets values to zeroes.
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

    if rampr > 0:
        right_sl = anyslice(array, slice(-rampr, None), dim)
    else:
        right_sl = anyslice(array, slice(None, None), dim)

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

def pad(array, dim, width, mode = 'edge', geometry = None):
    """
    Pad an array along a given dimension.
    numpy.pad seems to be very memory hungry! Don't use it for large arrays.

    Args:
        array: input array
        dim  : dim to apply the ramp
        width: width of the ramp
        mode :'linear' - creates linear decay of intensity; 'edge' - smears data in a costant manner; 'zero' - sets values to zeroes.
        geometry: geometry record to update (updates detector offset).
    """
    if min(width) < 0:
        raise Exception('Negative pad width found!')

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

    if padr == 0:
        sl = anyslice(new, slice(padl, None), dim)
    else:
        sl = anyslice(new, slice(padl,-padr), dim)

    new[sl] = array

    new = ramp(new, dim, width, mode)

    # Correct geometry if needed:
    if geometry:

        dicti = ['det_ort', 'det_tan', 'det_tan']
        offset = (padr - padl) / 2

        geometry[dicti[dim]] += offset * geometry['det_pixel']

    # If input is memmap - update it's size, release RAM memory.
    return rewrite_memmap(array, new)

def bin(array, dim = None, geometry = None):
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
             if geometry:
                 geometry.properties['img_pixel'] *= 2
                 geometry.properties['det_pixel'] *= 2

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

        array = array[:, :-1:2, :-1:2]

        for ii in range(array.shape[2]):
            array[:-1:2, :, ii] += array[1::2, :, ii]

        if geometry:
            geometry.properties['img_pixel'] *= 2
            geometry.properties['det_pixel'] *= 2

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

    if geometry:
        geometry['det_tan'] = geometry['det_tan']+ geometry.pixel[1] * h / 2
        geometry['det_ort'] = geometry['det_ort']+ geometry.pixel[0] * v / 2

    # Its better to leave the memmap file as it is. Return a view to it:
    return new

def cast2shape(array, shape):
    '''
    Make the array to conform with the given shape.
    '''
    if array.ndim != len(shape):
        raise Exception('Wrong array shape!')

    for ii in range(array.ndim):
        dif = array.shape[ii] - shape[ii]
        if dif > 0:
            wl = dif // 2
            wr = dif - wl
            array = crop(array, ii, [wl, wr])

        elif dif < 0:
            wl = -dif // 2
            wr = -dif - wl
            array = pad(array, ii, [wl, wr], mode = 'zero')

    return array

def flipdim(array, transpose = [1, 0, 2], updown = True):
    """
    Convert a given numpy array (sorted: index, hor, vert) to ASTRA-compatible projections stack
    """
    # Transpose seems to be compatible with memmaps:
    array = numpy.transpose(array, transpose)

    if updown:
        array = numpy.flipud(array)

    return array

def raw2astra(array):
    """
    Convert a given numpy array (sorted: index, hor, vert) to ASTRA-compatible projections stack
    """
    # Transpose seems to be compatible with memmaps:
    array = numpy.transpose(array, [1,0,2])
    array = numpy.flipud(array)

    return array.astype('float32')

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


def add_dim(array_1, array_2, dim = None):
    """
    Add two arrays with arbitrary dimensions. We assume that one or two dimensions match.
    """

    # Shapes to compare:
    shp1 = numpy.shape(array_1)
    shp2 = numpy.shape(array_2)

    dim1 = numpy.ndim(array_1)
    dim2 = numpy.ndim(array_2)

    if dim1 - dim2 == 0:
        array_1 += array_2

    elif dim1 - dim2 == 1:

        # Find dimension that is missing in array_2:
        if dim is None:
            dim = [ii not in shp2 for ii in shp1].index(True)

        if dim == 0:
            array_1 += array_2[None, :, :]
        elif dim == 1:
            array_1 += array_2[:, None, :]
        elif dim == 2:
            array_1 += array_2[:, :, None]

    elif dim1 - dim2 == 2:
        # Find dimension that is matching in array_2:
        if dim is None:
            dim = [ii in shp2 for ii in shp1].index(True)

        if dim == 0:
            array_1 += array_2[:, None, None]
        elif dim == 1:
            array_1 += array_2[None, :, None]
        else:
            array_1 += array_2[None, None, :]

    else:
        raise Exception('ERROR! array_1.ndim - array_2.ndim should be 0, 1 or 2')

def mult_dim(array_1, array_2, dim = None):
    """
    Multiply a 3D array by a 1D or a 2D vector along one of the dimensions.
    """
    # Shapes to compare:
    shp1 = numpy.shape(array_1)
    shp2 = numpy.shape(array_2)

    dim1 = numpy.ndim(array_1)
    dim2 = numpy.ndim(array_2)

    if dim1 - dim2 == 0:
        array_1 *= array_2

    elif dim1 - dim2 == 1:

        # Find dimension that is missing in array_2:
        if dim is None:
            dim = [ii not in shp2 for ii in shp1].index(True)

        if dim == 0:
            array_1 *= array_2[None, :, :]
        elif dim == 1:
            array_1 *= array_2[:, None, :]
        elif dim == 2:
            array_1 *= array_2[:, :, None]

    elif dim1 - dim2 == 2:

        # Find dimension that is matching in array_2:
        if dim is None:
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
