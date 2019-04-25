#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains acquisition geometry classes: circular, linear, helical.
The circular class corresponds to the simplest case of circular orbit cone-beam CT with minimal number of parameters.
Additional parameters can be use to define a non-conventional geometry.
For instance: offsets and rotations ('det_roll', 'det_tan', 'det_ort', etc.), axis tilts ('axs_roll', 'axs_pitch'),
volume transformations ('vol_tra', 'vol_rot'), recostruction resolution and anisotropic sampling ('img_pixel', 'det_sample', 'vol_sample').
"""
# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Imports >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

import astra

import numpy
from transforms3d import euler

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Classes >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class basic():
    '''
    Base geometry class. Needs only SDD, ODD and pixel size to initialize.
    Use 'parameters' to store parameters and 'description' to store relevant metadata.
    '''

    def __init__(self, src2obj = None, det2obj = None, det_pixel = None, img_pixel = None, unit = 'mm'):
        '''
        Constructor for the base geometry class.

        Args:
            src2obj  : source to detector distance
            det2obj  : object to detector distance
            det_pixel: detector pixel size
            img_pixel: reconstruction volume voxel size (optional)
            unit     : unit length (default = 'mm')
        '''
        # Default voxel size:
        if (not img_pixel) and (det_pixel): img_pixel = det_pixel / (src2obj + det2obj) * src2obj

        # Parameters:
        self.parameters = {'src2obj':src2obj,
                           'det2obj':det2obj,

                           'det_pixel':det_pixel,
                           'img_pixel':img_pixel,
                           'unit':unit,

                           'axs_tan':0,                 # rotation axis translation
                           'vol_rot':[0.0, 0.0, 0.0],   # volume rotations and translation vectors
                           'vol_tra':[0.0, 0.0, 0.0],

                           'det_sample':[1,1],          # detector and volume sampling (use to indicate that pixels were binned)
                           'vol_sample':[1,1,1]
                           }

        self.description = {}

    def __str__(self):
        '''
        Show my description.
        '''
        return 'Geometry class: ' + str(type(self)) + '\n Parameters: \n' + str(self.parameters)

    def __repr__(self):
        '''
        Show my description.
        '''
        return 'Geometry class: ' + str(type(self)) + '\n Parameters: \n' + str(self.parameters)

    def __getitem__(self, key):
        '''
        Retrieve one of the geometry parameters.
        '''
        val = self.parameters.get(key)

        if val is None:
            print('WARNING! A property with an unknown key is requested: ' + key)

        return val

    def __setitem__(self, key, val):
        '''
        Set one of the geometry parameters.
        '''
        if not key in self.parameters.keys():
            print('WARNING! A property with an unknown key is added to geometry: ' + key)

        self.parameters[key] = val

    def copy(self):
        '''
        Copy me.
        '''
        geom = type(self)()
        geom.parameters = self.parameters.copy()
        geom.description = self.description.copy()

        return geom

    def to_dictionary(self):
        '''
        Return a dictionary describing this geometry.
        '''
        records = self.parameters.copy()
        records.update(self.description)

        return records

    def from_matrix(self, R, T):
        """
        Rotates and translates the reconstruction volume.

        Args:
            R (3x3 array): rotation matrix
            T (1x3 array): translation vector
        """
        # Translate to flex geometry:
        self.parameters['vol_rot'] = numpy.rad2deg(euler.mat2euler(R.T, axes = 'sxyz'))
        self.parameters['vol_tra'] = numpy.array(self.parameters['vol_tra']) - numpy.dot(T, R.T)[[0,2,1]] * self.voxel

    def from_dictionary(self, dictionary):
        '''
        Use dictionary records to initialize this geometry.
        '''
        # Fill gaps if needed:
        if not dictionary.get('src2obj'):
            if (not dictionary.get('src2det')):
                raise Exception('Filed missing in geometry record: src2det')

            elif (not dictionary.get('det2obj')):
                raise Exception('Filed missing in geometry record: det2obj')

            dictionary['src2obj'] = dictionary['src2det'] - dictionary['det2obj']

        if (not dictionary.get('det2obj')):
            if (not dictionary.get('src2det')):
                raise Exception('Filed missing in geometry record: src2det')
            else:
                dictionary['det2obj'] = dictionary['src2det'] - dictionary['src2obj']

        # Copy records:
        for key in dictionary.keys():

            value = dictionary[key]

            if key in self.parameters.keys():
                self.parameters[key] = value

            else:
                self.description[key] = value

        # After-check:
        if not self.parameters.get('img_pixel'):
            self.parameters['img_pixel'] = self.parameters['det_pixel'] / self.magnification

        if not self.parameters.get('det_pixel'):
            self.parameters['det_pixel'] = self.parameters['img_pixel'] * self.magnification

        # Check if all necessary keys are there:
        min_set = ['src2obj', 'det2obj', 'det_pixel']
        for key in min_set:
            if not key in self.parameters:
                raise Exception('Geometry parameter missing after parcing: ' + key)

    @property
    def vol_sample(self):
        '''
        Voxel shape.
        '''
        return self.parameters.get('vol_sample')

    @vol_sample.setter
    def vol_sample(self, sample):
        '''
        Voxel shape.
        '''
        self.parameters['vol_sample'] = sample

    @property
    def det_sample(self):
        '''
        Pixel shape.
        '''
        return self.parameters.get('det_sample')

    @det_sample.setter
    def det_sample(self, sample):
        '''
        Pixel shape.
        '''
        self.parameters['det_sample'] = sample

    @property
    def pixel(self):
        '''
        Pixel size (mm).
        '''
        return self.parameters.get('det_pixel') * numpy.array(self.parameters.get('det_sample'))

    @property
    def voxel(self):
        '''
        Voxel size (mm).
        '''
        return self.parameters.get('img_pixel') * numpy.array(self.parameters.get('vol_sample'))

    @property
    def src2obj(self):
        '''
        Source-to-object distance.
        '''
        return self.parameters.get('src2obj')

    @property
    def det2obj(self):
        '''
        Detector-to-object distance.
        '''
        return self.parameters.get('det2obj')

    @property
    def src2det(self):
        '''
        Source-to-detector distance.
        '''
        return self.parameters.get('src2obj') + self.parameters.get('det2obj')

    @property
    def magnification(self):
        '''
        Magnification.
        '''
        return self.src2det / self.src2obj

    def volume_xyz(self, shape, offset = [0.,0.,0.]):
        """
        Coordinate grid in units of the geometry.

        Args:
            shape : volume shape
            offset: offset in length units
        """
        xx = (numpy.arange(0, shape[0]) - shape[0] / 2) * self.voxel[0] - offset[0]
        yy = (numpy.arange(0, shape[1]) - shape[1] / 2) * self.voxel[1] - offset[1]
        zz = (numpy.arange(0, shape[2]) - shape[2] / 2) * self.voxel[2] - offset[2]

        return xx, yy, zz

    def from_astra_cone_vec(self, vectors):
        '''
        Take vectors from ASTRA 'cone_vec' geometry. This will override any other parameters.
        '''
        self._vectors_ = vectors

    def astra_projection_geom(self, data_shape, index = None):
        '''
        Get ASTRA projection geometry.

        Args:
            data_shape: [detector_count_z, theta_count, detector_count_x]
            index     : if provided - sequence of the rotation angles

        Returns:
            geometry : ASTRA cone-beam geometry.
        '''
        # Get vectors:
        if hasattr(self, '_vectors_'):
            vectors = self._vectors_

        else:
            vectors = self.get_vectors(data_shape[1], index)

        # Get ASTRA geometry:
        det_count_x = data_shape[2]
        det_count_z = data_shape[0]
        return astra.create_proj_geom('cone_vec', det_count_z, det_count_x, vectors)

    def astra_volume_geom(self, vol_shape, slice_first = None, slice_last = None):
        '''
        Initialize ASTRA volume geometry.

        Args:
            vol_shape  : volume array shape
            slice_first: first slice of an ROI to update
            slice_last : last slice of an ROI to update
        '''

        # Shape and size (mm) of the volume
        vol_shape = numpy.array(vol_shape)
        vol_size = vol_shape * self.voxel

        if (slice_first is not None) & (slice_last is not None):
            # Generate volume geometry for one chunk of data:

            length = vol_shape[0]

            # Compute offset from the centre:
            centre = (length - 1) / 2
            offset = (slice_first + slice_last) / 2 - centre
            offset = offset * self.voxel[0]

            shape = [slice_last - slice_first + 1, vol_shape[1], vol_shape[2]]
            vol_size = numpy.array(shape) * self.voxel[0]

        else:
            shape = vol_shape
            offset = 0

        #vol_geom = astra.creators.create_vol_geom(shape[1], shape[2], shape[0],
        vol_geom = astra.create_vol_geom(shape[1], shape[2], shape[0],
                  -vol_size[2]/2, vol_size[2]/2, -vol_size[1]/2, vol_size[1]/2,
                  -vol_size[0]/2 + offset, vol_size[0]/2 + offset)

        return vol_geom

    def get_vectors(self, proj_count, index = None):
        '''
        Get source, detector and detector orientation vectors.

        Args:
            angle_count : number of rotation angles
            index       : index of angles that should be used
        '''
        # Create source orbit:
        src_vect = self.get_source_orbit(proj_count, index)

        # Create detector orbit (same as source but 180 degrees offset):
        det_vect, det_tan, det_rad, det_orth = self.get_detector_orbit(proj_count, index)

        # Apply global rotations and translations:
        self._transform_vectors_(src_vect, det_vect, det_tan, det_rad, det_orth)

        # Append all vectors together:
        vectors = numpy.concatenate([src_vect, det_vect, det_tan * self.pixel[1], det_orth * self.pixel[0]], axis = 1)
        return vectors

    def get_source_orbit(self, proj_count = None, index = None):
        '''
        Get the source orbit. In the base class it is a circular orbit.

        Args:
            proj_count: number of projections
            index           : index of the projection subset

        Returns:
            src_pos : array of the source positions. Can be generated by circular_orbit(...), for instance.
        '''
        raise Exception('Override this method in a geometry class derived from the base class!')

    def get_detector_orbit(self, proj_count = None, index = None):
        '''
        Get the detector orbit. In the base class it is a circular orbit.

        Args:
            proj_count: number of projections
            index           : index of the projection subset
        Returns:
            src_pos : array of the source positions. Can be generated by circular_orbit(...), for instance.
        '''
        raise Exception('Override this method in a geometry class derived from the base class!')

    def _transform_vectors_(self, src_vect, det_vect, det_tan, det_rad, det_orth):
        '''
        Rotate and translate vectors depending on the volume orientation.
        '''
        vol_rot = self.parameters['vol_rot']
        vol_tra = self.parameters['vol_tra']

        # Rotate everything relative to the reconstruction volume:
        R = _euler2mat_(vol_rot[0], vol_rot[1], vol_rot[2], 'rzyx')
        det_tan[:] = numpy.dot(det_tan, R)
        det_rad[:] = numpy.dot(det_rad, R)
        det_orth[:] = numpy.dot(det_orth, R)
        src_vect[:] = numpy.dot(src_vect,R)
        det_vect[:] = numpy.dot(det_vect,R)

        # Add translation:
        T = numpy.array([vol_tra[1], vol_tra[2], vol_tra[0]])
        src_vect -= numpy.dot(T, R)
        det_vect -= numpy.dot(T, R)

    def detector_size(self, proj_shape):
        '''
        Get the size of detector in length units.
        '''
        if len(proj_shape) == 3:
            return numpy.array(proj_shape[::2]) * self.pixel
        else:
            return numpy.array(proj_shape) * self.pixel

    def detector_centre(self):
        '''
        Get the centre coordinate of the first position of the detector.
        '''
        det_pos, det_tan, det_rad, det_orth = self.get_detector_orbit(proj_count = 3)

        return [det_pos[0][2], det_pos[0][0]]

    def detector_bounds(self, proj_shape):
        '''
        Get the boundaries of the detector at the start of the scan in length units.
        '''

        det_pos, det_tan, det_rad, det_orth = self.get_detector_orbit(proj_count = 3)

        sz = self.detector_size(proj_shape) / 2
        cntr = self.detector_centre()

        vrt = [-sz[0], sz[0]]
        hrz = [-sz[1], sz[1]]

        return numpy.array([vrt, hrz]) + numpy.array(cntr)[:, None]

    def volume_size(self, vol_shape):
        '''
        Return volume size in length units.
        '''
        return numpy.array(vol_shape) * self.voxel

    def volume_bounds(self, vol_shape):
        '''
        Return volume bounds:
        '''
        sz = self.volume_size(vol_shape) / 2

        vrt = [-sz[0], sz[0]]
        mag = [-sz[1], sz[1]]
        hrz = [-sz[2], sz[2]]

        return numpy.array([vrt, mag, hrz]) + numpy.array(self.parameters['vol_tra'])[:, None]

class circular(basic):
    '''
    Circular orbit geometry class. Includes additional parameters such as detector and source shifts and rotations.
    '''

    def __init__(self, src2obj = None, det2obj = None, det_pixel = None, img_pixel = None, ang_range = (0, 360), unit = 'mm'):
        '''
        Constructor for the circular geometry class.

        Args:
            src2obj  : source to detector distance
            det2obj  : object to detector distance
            det_pixel: detector pixel size
            img_pixel: reconstruction volume voxel size (optional)
            ang_range: range of rotation (default = 0..360)
            unit     : unit length (default = 'mm')
        '''
        # Parent init:
        basic.__init__(self, src2obj, det2obj, det_pixel, img_pixel, unit)

        # Additional parameters:
        self.parameters.update({
                           'ang_range': ang_range,
                           'src_ort':0,   # source vertical, tangential shifts
                           'src_tan':0,

                           'det_ort':0,   # detector shifts and rotations (in degrees)
                           'det_tan':0,
                           'det_roll':0,
                           'det_pitch':0,
                           'det_yaw':0,

                           'axs_roll':0,  # rotation axis roll and pitch (in degrees)
                           'axs_pitch':0,
                           })

    def get_thetas(self, proj_count = None, index = None):
        '''
        Get rotation angles. Either returns an equidistant array or self.thetas if that is defined.

        Args:
            proj_count : number of angles in equidistant case. Not used if 'thetas' are defined explicitly in parameters.

        Returns:
            thetas : angle array in degrees
        '''
        thetas = self.parameters.get('thetas')

        # Initialize thetas:
        if thetas is None:
            thetas = numpy.linspace(self['ang_range'][0], self['ang_range'][1], proj_count)

        if not index is None: thetas = thetas[index]

        return thetas

    def get_source_orbit(self, proj_count = None, index = None):
        '''
        Get the source orbit.

        Args:
            proj_count: number of projections
            index     : index of the projection subset

        Returns:
            src_pos : array of the source positions.
        '''
        src2obj = self.src2obj

        src_ort = self.parameters['src_ort']
        src_tan = self.parameters['src_tan']

        axs_tan = self.parameters['axs_tan']

        axs_roll = self.parameters['axs_roll']
        axs_pitch = self.parameters['axs_pitch']

        # Create source orbit:
        thetas = self.get_thetas(proj_count)

        # src_ort may be a vector or a scalar:
        org = numpy.outer(src_ort, [0,0,1])

        src_vect, src_tan, src_rad, serc_orth = circular_orbit(src2obj, thetas, roll = axs_roll, pitch = axs_pitch, yaw = 0,
                                                         origin = org, tan_shift = src_tan - axs_tan, index = index)

        return src_vect

    def get_detector_orbit(self, proj_count = None, index = None):
        '''
        Get the detector orbit.

        Args:
            proj_count: number of projections
            index           : index of the projection subset

        Returns:
            det_pos : array of the detector positions.
            det_tan : array of detector tangential directional vector
            det_rad : array of detector radial directional vector
            det_orth: array of detector orthogonal directional vector
        '''
        det2obj = self.det2obj

        # Detector translations:
        det_ort = self.parameters['det_ort']
        det_tan = self.parameters['det_tan']

        # Rotation axis translations and rotations:
        axs_tan = self.parameters['axs_tan']

        axs_roll = self.parameters['axs_roll']
        axs_pitch = self.parameters['axs_pitch']

        # Detector rotations:
        det_roll = self.parameters['det_roll']
        det_yaw = self.parameters['det_yaw']
        det_pitch = self.parameters['det_pitch']

        # Create detector orbit:
        thetas = self.get_thetas(proj_count)

        # det_ort may be a vector or a scalar:
        org = numpy.outer(det_ort, [0,0,1])

        det_pos, det_tan, det_rad, det_orth = circular_orbit(det2obj, thetas, roll = axs_roll, pitch = axs_pitch, yaw = 180,
                                                         origin = org, tan_shift = -det_tan + axs_tan, index = index)

        # Invert vectors to keep them alligned with the source vectors:
        det_tan, det_rad, det_orth = -det_tan, -det_rad, -det_orth

        # Apply detector rotations:
        for ii in range(det_pos.shape[0]):

            T = _axangle2mat_(det_rad[ii, :], det_roll)
            det_tan[ii, :] = T.dot(det_tan[ii, :])
            det_orth[ii, :] = T.dot(det_orth[ii, :])

            T = _axangle2mat_(det_orth[ii, :], det_yaw)
            det_tan[ii, :] = T.dot(det_tan[ii, :])
            det_rad[ii, :] = T.dot(det_rad[ii, :])

            T = _axangle2mat_(det_tan[ii, :], det_pitch)
            det_rad[ii, :] = T.dot(det_rad[ii, :])
            det_orth[ii, :] = T.dot(det_orth[ii, :])

        return det_pos, det_tan, det_rad, det_orth

class helical(circular):
    '''
    Helical orbit geometry class. Similar to the 'circular' class with additional parameter of helix
    '''

    def __init__(self, src2obj = None, det2obj = None, det_pixel = None, img_pixel = None, axis_range = (0, 100),  ang_range = (0, 720), unit = 'mm'):
        '''
        Constructor for the helical geometry class.

        Args:
            src2obj  : source to detector distance
            det2obj  : object to detector distance
            det_pixel: detector pixel size
            img_pixel: reconstruction volume voxel size (optional)
            axis_range: range of the movement along the axis of rotation
            ang_range: range of angles (default = 0..360)
            unit     : unit length (default = 'mm')
        '''
        # Parent init:
        circular.__init__(self, src2obj, det2obj, det_pixel, img_pixel, ang_range, unit)

        # Additional parameters:
        self.parameters['axs_rng'] = axis_range

    def get_source_orbit(self, proj_count = None, index = None):
        '''
        Get the source orbit. In the base class it is a circular orbit.
        '''
        src2obj = self.src2obj

        src_ort = self.parameters['src_ort']
        src_tan = self.parameters['src_tan']

        axs_tan = self.parameters['axs_tan']

        axs_roll = self.parameters['axs_roll']
        axs_pitch = self.parameters['axs_pitch']

        # Create source orbit:
        thetas = self.get_thetas(proj_count)

        # src_ort may be a vector or a scalar:
        org = numpy.outer(src_ort, [0,0,1])

        src_vect, src_tan, src_rad, src_orth = circular_orbit(src2obj, thetas, roll = axs_roll, pitch = axs_pitch, yaw = 0,
                                                         origin = org, tan_shift = src_tan - axs_tan, index = index)

        # Add axial motion:
        vrt = numpy.linspace(self.parameters['axs_rng'][0], self.parameters['axs_rng'][1], proj_count)
        if index: vrt = vrt[index]
        src_vect = src_vect + src_orth * vrt[:, None]

        return src_vect

    def get_detector_orbit(self, proj_count = None, index = None):
        '''
        Get the detector orbit. In the base class it is a circular orbit.
        '''
        det2obj = self.det2obj

        # Detector translations:
        det_ort = self.parameters['det_ort']
        det_tan = self.parameters['det_tan']

        # Rotation axis translations and rotations:
        axs_tan = self.parameters['axs_tan']

        axs_roll = self.parameters['axs_roll']
        axs_pitch = self.parameters['axs_pitch']

        # Detector rotations:
        det_roll = self.parameters['det_roll']
        det_yaw = self.parameters['det_yaw']
        det_pitch = self.parameters['det_pitch']

        # Create detector orbit:
        thetas = self.get_thetas(proj_count)

        # det_ort may be a vector or a scalar:
        org = numpy.outer(det_ort, [0,0,1])

        det_pos, det_tan, det_rad, det_orth = circular_orbit(det2obj, thetas, roll = axs_roll, pitch = axs_pitch, yaw = 180,
                                                         origin = org, tan_shift = -det_tan + axs_tan, index = index)

        # Add axial motion:
        vrt = numpy.linspace(self.parameters['axs_rng'][0], self.parameters['axs_rng'][1], proj_count)
        if index: vrt = vrt[index]
        det_pos = det_pos + det_orth * vrt[:, None]

        # Invert vectors to keep them alligned with the source vectors:
        det_tan, det_rad, det_orth = -det_tan, -det_rad, -det_orth

        # Apply detector rotations:
        for ii in range(det_pos.shape[0]):

            T = _axangle2mat_(det_rad[ii, :], det_roll)
            det_tan[ii, :] = T.dot(det_tan[ii, :])
            det_orth[ii, :] = T.dot(det_orth[ii, :])

            T = _axangle2mat_(det_orth[ii, :], det_yaw)
            det_tan[ii, :] = T.dot(det_tan[ii, :])
            det_rad[ii, :] = T.dot(det_rad[ii, :])

            T = _axangle2mat_(det_tan[ii, :], det_pitch)
            det_rad[ii, :] = T.dot(det_rad[ii, :])
            det_orth[ii, :] = T.dot(det_orth[ii, :])

        return det_pos, det_tan, det_rad, det_orth

class linear(basic):
    '''
    A simple linear orbit geometry class.
    '''

    def __init__(self, src2obj = None, det2obj = None, det_pixel = None, img_pixel = None,
                 src_hrz_rng = (0, 1), src_vrt_rng = (0, 1), det_hrz_rng = (1, 0), det_vrt_rng = (1, 0), unit = 'mm'):
        '''
        Constructor for the linear geometry class.

        Args:
            src2obj  : source to detector distance
            det2obj  : object to detector distance
            det_pixel: detector pixel size
            img_pixel: reconstruction volume voxel size (optional)
            src_hrz_rng: source horizlontal movement range
            src_vrt_rng: source vertical movement range
            det_hrz_rng: detector horizlontal movement range
            det_vrt_rng: detector vertical movement range
            unit     : unit length (default = 'mm')
        '''
        # Parent init:
        basic.__init__(self, src2obj, det2obj, det_pixel, img_pixel, unit)

        # Additional parameters:
        self.parameters.update({
                           'src_hrz_rng':src_hrz_rng,   # source vertical, horizlontal motion range
                           'src_vrt_rng':src_vrt_rng,

                           'det_hrz_rng':det_hrz_rng,   # source vertical, horizlontal motion range
                           'det_vrt_rng':det_vrt_rng,

                           'det_roll':0,               # detector rotations (in degrees)
                           'det_pitch':0,
                           'det_yaw':0,
                           })

    def get_source_orbit(self, proj_count = None, index = None):
        '''
        Get the source orbit. In the base class it is a circular orbit.
        '''
        src2obj = self.src2obj

        src_hrz_rng = self.parameters['src_hrz_rng']
        src_vrt_rng = self.parameters['src_vrt_rng']

        # Create source orbit:
        src_vect, src_tan, src_rad, serc_ort = linear_orbit(src_hrz_rng, (-src2obj, -src2obj), src_vrt_rng, proj_count, index)

        return src_vect

    def get_detector_orbit(self, proj_count = None, index = None):
        '''
        Get the detector orbit. In the base class it is a circular orbit.
        '''
        det2obj = self.det2obj

        det_hrz_rng = self.parameters['det_hrz_rng']
        det_vrt_rng = self.parameters['det_vrt_rng']

        # Detector rotations:
        det_roll = self.parameters['det_roll']
        det_yaw = self.parameters['det_yaw']
        det_pitch = self.parameters['det_pitch']

        # Create detector orbit:
        det_pos, det_tan, det_rad, det_orth = linear_orbit(det_hrz_rng, (det2obj, det2obj), det_vrt_rng, proj_count, index)

        # Apply detector rotations:
        for ii in range(det_pos.shape[0]):

            T = _axangle2mat_(det_rad[ii, :], det_roll)
            det_tan[ii, :] = T.dot(det_tan[ii, :])
            det_orth[ii, :] = T.dot(det_orth[ii, :])

            T = _axangle2mat_(det_orth[ii, :], det_yaw)
            det_tan[ii, :] = T.dot(det_tan[ii, :])
            det_rad[ii, :] = T.dot(det_rad[ii, :])

            T = _axangle2mat_(det_tan[ii, :], det_pitch)
            det_rad[ii, :] = T.dot(det_rad[ii, :])
            det_orth[ii, :] = T.dot(det_orth[ii, :])

        return det_pos, det_tan, det_rad, det_orth

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Static methods >>>>>>>>>>>>>>>>>>>>>>>
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

    det_pixel = geometry_list[0].pixel

    axs_hrz = 0

    # Find out the size required for the final dataset
    for geo in geometry_list:

        bounds = geo.detector_bounds(shape)

        min_x = min([min_x, bounds[1][0]])
        min_y = min([min_y, bounds[0][0]])
        max_x = max([max_x, bounds[1][1]])
        max_y = max([max_y, bounds[0][1]])

        axs_hrz += geo['axs_tan'] / len(geometry_list)

    # Big slice:
    new_shape = numpy.array([(max_y - min_y) / det_pixel[0], shape[1], (max_x - min_x) / det_pixel[1]])
    new_shape = numpy.ceil(new_shape).astype('int') # safety margin..

    # Copy one of the geometry records and sett the correct translation:
    geometry = geometry_list[0].copy()

    geometry['axs_tan'] = axs_hrz

    geometry['det_tan'] = (max_x + min_x) / 2 + axs_hrz
    geometry['det_ort'] = (max_y + min_y) / 2

    # Update volume center:
    geometry['vol_tra'][0] = (geometry['det_ort'] * geometry.src2obj + geometry['src_ort'] * geometry.det2obj) / geometry.src2det
    geometry['vol_tra'][2] = axs_hrz

    return new_shape, geometry

def astra_projection_geom(geom, data_shape, index = None):
    '''
    Initialize ASTRA projection geometry.

    Args:
        geom      : geometry class
        data_shape: [detector_count_z, theta_count, detector_count_x]
        index     : if provided - sequence of the rotation angles
    '''

    return geom.astra_projection_geom(data_shape, index)


def astra_volume_geom(geom, vol_shape, slice_first = None, slice_last = None):
    '''
    Initialize ASTRA volume geometry.
    '''
    return geom.astra_volume_geom(vol_shape, slice_first = None, slice_last = None)

def get_vectors(geom, angle_count, index = None):
    '''
    Get source, detector and detector orientation vectors.

    Args:
        geom       : geometry class
        angle_count : number of rotation angles
        index       : index of angles that should be used
    '''
    return geom.get_vectors(angle_count, index)

def detector_size(geom, proj_shape):
    '''
    Get the size of detector in length units.
    '''
    return geom.detector_size(proj_shape)

def detector_bounds(geom, proj_shape):
    '''
    Get the boundaries of the detector in length units.
    '''
    return geom.detector_bounds(proj_shape)

def volume_bounds(geom, proj_shape):
    '''
    A very simplified version of volume bounds...
    '''
    return geom.volume_bounds(proj_shape)

def volume_shape(geom, proj_shape):
    '''
    Based on physical volume bnounds compute shape in pixels:
    '''
    return geom.volume_shape(proj_shape)


def linear_orbit(hrz_rng, rad_rng, vrt_rng, proj_count, index = None):
    '''
    Generate a linear orbit vector.

    Args:
        hrz_rng: horizontal range of motion
        rad_rng: radial range of motion
        vrt_rng: vertical range of motion

    Returns:
        position   : position vector
        tangent    : tangent direction
        radius     : radal direction
        orthogonal : orthogonal direction
    '''

    h = numpy.linspace(hrz_rng[0], hrz_rng[1], proj_count)
    r = numpy.linspace(rad_rng[0], rad_rng[1], proj_count)
    v = numpy.linspace(vrt_rng[0], vrt_rng[1], proj_count)

    position = numpy.stack((h, r, v)).transpose((1,0))

    radius = numpy.zeros([proj_count, 3])
    radius[:, 1] = 1

    tangent = numpy.zeros([proj_count, 3])
    tangent[:, 0] = 1

    # Orthogonal to tangent and radius:
    orthogonal = numpy.cross(radius, tangent)

    if index is not None:
        return position[index], tangent[index], radius[index], orthogonal[index]
    else:
        return position, tangent, radius, orthogonal

def circular_orbit(radius, thetas, roll = 0, pitch = 0, yaw = 0,
                     origin = [0, 0, 0], tan_shift = 0, index = None):
    '''
    Generate a circular orbit vector.

    Args:
        radius     : orbit radius
        angle_count: number of rotation angles
        roll, pitch: define orientation of the rotation axis
        yaw        : initial angular position
        origin     : xyz vector of the orbit centre
        tan_shift  : tangential shift from the default position (scalar or array)
        index      : index of the subset of total rotation angles

    Returns:
        position   : position vector
        tangent    : tangent direction
        radius     : radal direction
        orthogonal : orthogonal direction
    '''
    # Generate axis and orthogonals:
    M = _euler2mat_(pitch, roll, 0)
    axis = M.dot([0, 0, 1])
    tan0 = M.dot([1, 0, 0])
    rad0 = M.dot([0, 1, 0])

    # Genertate initial circular orbit:
    M = _axangle2mat_(axis, yaw)
    v0 = M.dot([0, -radius, 0])
    tan0 = M.dot(tan0)
    rad0 = M.dot(rad0)

    position = numpy.zeros([len(thetas), 3])
    tangent = numpy.zeros([len(thetas), 3])
    radius = numpy.zeros([len(thetas), 3])

    for ii, theta in enumerate(thetas):
        Rt = _axangle2mat_(axis, theta)
        position[ii, :] = Rt.dot(v0)
        tangent[ii, :] = Rt.dot(tan0)
        radius[ii, :] = Rt.dot(rad0)

    # Apply origin shift:
    if numpy.ndim(origin) == 1:
        position += numpy.array(origin)[None, :]
    else:
        position += numpy.array(origin)

    # Apply other shifts:
    if numpy.size(tan_shift) == 1:
        position += tangent * tan_shift
    else:
        position += tangent * tan_shift[:, None]

    # Orthogonal to tangent and radius:
    orthogonal = numpy.cross(radius, tangent)

    if index is not None:
        return position[index], tangent[index], radius[index], orthogonal[index]
    else:
        return position, tangent, radius, orthogonal

def _euler2mat_(a, b, c, axes='sxyz'):
    a = numpy.deg2rad(a)
    b = numpy.deg2rad(b)
    c = numpy.deg2rad(c)

    return euler.euler2mat(a, b, c, axes)

def _axangle2mat_(ax, a):
    a = numpy.deg2rad(a)

    return euler.axangle2mat(ax, a)
