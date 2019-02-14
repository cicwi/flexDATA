#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kostenko
Created on Feb 2019

This module contains acquisition geometry classes.
"""
# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Imports >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

import astra

import numpy
from transforms3d import euler

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
    
    det_pixel = geometry_list[0]['det_pixel']
    
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
    new_shape = numpy.array([(max_y - min_y) / det_pixel, shape[1], (max_x - min_x) / det_pixel])                     
    new_shape = numpy.round(new_shape).astype('int')
    
    # Copy one of the geometry records and sett the correct translation:
    geometry = geometry_list[0].copy()
    
    geometry['axs_tan'] = axs_hrz
    
    geometry['det_tan'] = (max_x + min_x) / 2
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

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Classes >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class basic():
    '''
    Basic geometry class. Needs only SDD, ODD and pixel size to initialize.
    '''
    
    def __init__(self, src2obj = None, det2obj = None, det_pixel = None, img_pixel = None, ang_range = (0, 360), unit = 'mm'):
        '''
        Constructor for the basic geometry class.
        
        Args:
            src2obj  : source to detector distance
            det2obj  : object to detector distance
            det_pixel: detector pixel size
            img_pixel: reconstruction volume voxel size (optional)
            ang_range: range of angles (default = 0..360)
            unit     : unit length (default = 'mm')
        '''
        # Default voxel size:
        if (not img_pixel) and (det_pixel): img_pixel = det_pixel / (src2obj + det2obj) * src2obj
        
        # Parameters:
        self.parameters = {'src2obj':src2obj, # basic parameters:
                           'det2obj':det2obj,
                           'det_pixel':det_pixel,
                           'img_pixel':img_pixel,
                           'unit':unit,
                           'range':ang_range,
                           
                           'vol_rot':[0.0, 0.0, 0.0],   # volume rotations and translation vectors
                           'vol_tra':[0.0, 0.0, 0.0],
                           
                           'det_sample':[1,1,1],  # detector and volume sampling
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
        return self.parameters.get(key)
    
    def __setitem__(self, key, val):
        '''
        Set one of the geometry parameters.
        '''
        if not key in self.parameters.keys():
            print('WARNING! A propert with an unknown key is added to geometry: ' + key)
            
        self.parameters[key] = val
        
    def copy(self):
        
        geom = type(self)()
        geom.parameters = self.parameters.copy()
        geom.description = self.description.copy()
        
        return geom
    
    def to_dictionary(self):
        '''
        Return a dictionary describing the geometry.
        '''
        records = self.parameters.copy()
        records.update(self.description)
        
        return records
        
        
    def from_dictionary(self, dictionary):
        '''
        Use dictionary records to initialize.
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
        min_set = ['src2obj', 'det2obj', 'det_pixel', 'range']
        for key in min_set:
            if not key in self.parameters:
                raise Exception('Geometry parameter missing after parcing: ' + key)    
            
    @property    
    def vol_sample(self):
        return self.parameters.get('vol_sample')
    
    @property    
    def det_sample(self):
        return self.parameters.get('det_sample')
    
    @property    
    def pixel(self):
        return self.parameters.get('det_pixel') * numpy.array(self.parameters.get('det_sample'))
    
    @property    
    def voxel(self):
        return self.parameters.get('img_pixel') * numpy.array(self.parameters.get('vol_sample'))
    
    @property    
    def range(self):
        return self.parameters.get('range')
    
    @property    
    def src2obj(self):
        return self.parameters.get('src2obj')
    
    @property    
    def det2obj(self):
        return self.parameters.get('det2obj')
    
    @property    
    def src2det(self):
        return self.parameters.get('src2obj') + self.parameters.get('det2obj')

    @property    
    def magnification(self):
        return self.src2det / self.src2obj
    
    def volume_xyz(self, shape, offset = [0.,0.,0.]):
        """
        Coordinate space in units of the geometry.
        """    
        xx = (numpy.arange(0, shape[0]) - shape[0] / 2) * self.voxel[0] - offset[0] 
        yy = (numpy.arange(0, shape[1]) - shape[1] / 2) * self.voxel[1] - offset[1]
        zz = (numpy.arange(0, shape[2]) - shape[2] / 2) * self.voxel[2] - offset[2]
        
        return xx, yy, zz
    
    def get_thetas(self, angle_count = None, index = None):
        '''
        Get rotation angles. Either returns an equidistant array or self.thetas if that is defined.
        
        Args:
            angle_count : number of angles. Not used if 'thetas' are defined explicitly in parameters.
        '''
        
        thetas = self.parameters.get('thetas')
        
        # Initialize thetas:
        if thetas is None:
            thetas = numpy.linspace(self.range[0], self.range[1], angle_count)
            
        if not index is None: thetas = thetas[index]    
        
        return thetas
    
    def astra_projection_geom(self, data_shape, index = None):
        '''
        Initialize ASTRA projection geometry.        
        
        Args:
            data_shape: [detector_count_z, theta_count, detector_count_x]
            index     : if provided - sequence of the rotation angles
        '''  
        # Get vectors: 
        ang_count = data_shape[1]                
        vectors = self.get_vectors(ang_count, index)
                
        # Get ASTRA geometry:
        det_count_x = data_shape[2]
        det_count_z = data_shape[0]
        return astra.create_proj_geom('cone_vec', det_count_z, det_count_x, vectors)
    
    
    def astra_volume_geom(self, vol_shape, slice_first = None, slice_last = None):
        '''
        Initialize ASTRA volume geometry.        
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
            vol_size = shape * self.voxel[0]
    
        else:
            shape = vol_shape
            offset = 0     
            
        #vol_geom = astra.creators.create_vol_geom(shape[1], shape[2], shape[0], 
        vol_geom = astra.create_vol_geom(shape[1], shape[2], shape[0], 
                  -vol_size[2]/2, vol_size[2]/2, -vol_size[1]/2, vol_size[1]/2, 
                  -vol_size[0]/2 + offset, vol_size[0]/2 + offset)
            
        return vol_geom              
    
    def get_vectors(self, angle_count, index = None):
        '''
        Get source, detector and detector orientation vectors.
        
        Args:
            angle_count : number of rotation angles
            index       : index of angles that should be used
        '''        
        # Create source orbit:
        src_vect = self._get_source_orbit_(angle_count, index)
        
        # Create detector orbit (same as source but 180 degrees offset):
        det_vect, det_tan, det_rad, det_orth = self._get_detector_orbit_(angle_count, index)
             
        # Apply global rotations and translations:                      
        self._global_transform_(src_vect, det_vect, det_tan, det_rad, det_orth)
        
        # Append all vectors together:
        vectors = numpy.concatenate([src_vect, det_vect, det_tan * self.pixel[2], det_orth * self.pixel[0]], axis = 1)        
        return vectors  
    
    def detector_size(self, proj_shape):
        '''
        Get the size of detector in length units.
        '''  
        return numpy.array(proj_shape) * self['det_pixel']
    
    def detector_bounds(self, proj_shape):
        '''
        Get the boundaries of the detector in length units.
        '''      	
        sz = self.detector_size(proj_shape) / 2
        
        vrt = [-sz[0], sz[0]]
        hrz = [-sz[2], sz[2]]
        
        return numpy.array([vrt, hrz])

    def volume_bounds(self, proj_shape):
        '''
        A very simplified version of volume bounds...
        '''
        # TODO: Compute this propoerly.... Dont trust the horizontal bounds!!!
        
        # Detector bounds:
        det_bounds = self.detector_bounds(proj_shape)
        
        vrt = det_bounds[0]
        hrz = det_bounds[1]
        
        vrt_bounds = vrt
        hrz_bounds = [self['vol_tra'][2] + hrz[0], self['vol_tra'][2] + hrz[1]]
        mag_bounds = [self['vol_tra'][1] + hrz[0], self['vol_tra'][1] + hrz[1]]
                        
        return numpy.array([vrt_bounds, mag_bounds, hrz_bounds])
    
    def volume_shape(self, proj_shape):
        '''
        Based on physical volume bnounds compute shape in pixels:
        '''
        bounds = self.volume_bounds(proj_shape)
            
        range_vrt = numpy.ceil(bounds[0] / self.voxel[0])
        range_hrz = numpy.ceil(bounds[2] / self.voxel[2])
        range_mag = numpy.ceil(bounds[1] / self.voxel[1])
        
        range_vrt = range_vrt[1] - range_vrt[0]
        range_hrz = range_hrz[1] - range_hrz[0]
        range_mag = range_mag[1] - range_mag[0]
        
        return numpy.int32([range_vrt, range_mag, range_hrz])
    
    def _global_transform_(self, src_vect, det_vect, det_tan, det_rad, det_orth):
        '''
        Rotate and translate vectors depending on the volume orientation.
        '''
        vol_rot = self.parameters['vol_rot']
        vol_tra = self.parameters['vol_tra']
        
        # Rotate everything relative to the reconstruction volume:
        R = self._euler2mat_(vol_rot[0], vol_rot[1], vol_rot[2], 'rzyx')
        det_tan[:] = numpy.dot(det_tan, R)
        det_rad[:] = numpy.dot(det_rad, R)
        src_vect[:] = numpy.dot(src_vect,R)
        det_vect[:] = numpy.dot(det_vect,R)            
                
        # Add translation:
        T = numpy.array([vol_tra[1], vol_tra[2], vol_tra[0]])    
        src_vect -= numpy.dot(T, R)          
        det_vect -= numpy.dot(T, R)
        
    def _get_source_orbit_(self, angle_count = None, index = None):    
        '''
        Get the source orbit. In the base class it is a circular orbit.
        '''
        src2obj = self.src2obj
                        
        # Create source orbit:
        src_vect, src_tan, src_rad, serc_orth = self._circular_orbit_(src2obj, angle_count, roll = 0, pitch = 0, yaw = 0, 
                                                         origin = [0, 0, 0], tan_shift = 0, index = index)
        
        return src_vect
    
    def _get_detector_orbit_(self, angle_count = None, index = None):    
        '''
        Get the detector orbit. In the base class it is a circular orbit.
        '''
        det2obj = self.det2obj
                       
        # Create detector orbit:
        det_pos, det_tan, det_rad, det_orth = self._circular_orbit_(det2obj, angle_count, roll = 0, pitch = 0, yaw = 180, 
                                                         origin = [0, 0, 0], tan_shift =  0, index = index)
        
        # Invert vectors to keep them alligned with the source vectors:
        det_tan, det_rad, det_orth = -det_tan, -det_rad, -det_orth
                
        return det_pos, det_tan, det_rad, det_orth
    
    def _circular_orbit_(self, radius, angle_count, roll = 0, pitch = 0, yaw = 0,
                         origin = [0, 0, 0], tan_shift = 0, index = None):
        '''
        Generate a circular orbit vector.
        
        Args:
            radius     : orbit radius
            angle_count: number of rotation angles
            roll, pitch: define orientation of the rotation axis
            yaw        : initial angular position
            origin     : xyz vector of the orbit centre
            tan_shift  : tangential shift from the default position
            index      : index of the subset of total rotation angles
            
        Returns:
            position   : position vector
            tangent    : tangent direction
            radius     : radal direction
            orthogonal : orthogonal direction
        '''
        # THeta vector:
        thetas = self.get_thetas(angle_count, index)
        
        # Generate axis and orthogonals:
        axis = self._euler2mat_(roll, pitch, 0).dot([0, 0, 1])
        tan0 = self._euler2mat_(roll, pitch, 0).dot([1, 0, 0])
        rad0 = self._euler2mat_(roll, pitch, 0).dot([0, 1, 0])
        
        # Genertate initial circular orbit:
        v0 = self._axangle2mat_(axis, yaw).dot([0, -radius, 0])
        tan0 = self._axangle2mat_(axis, yaw).dot(tan0)
        rad0 = self._axangle2mat_(axis, yaw).dot(rad0)
        
        position = numpy.zeros([len(thetas), 3])
        tangent = numpy.zeros([len(thetas), 3])
        radius = numpy.zeros([len(thetas), 3])
        
        for ii, theta in enumerate(thetas):
            Rt = self._axangle2mat_(axis, theta)
            position[ii, :] = Rt.dot(v0)
            tangent[ii, :] = Rt.dot(tan0)
            radius[ii, :] = Rt.dot(rad0)
            
        # Apply origin shift:
        position += numpy.array(origin)[None, :]
        
        # Apply other shifts:
        position += tangent * tan_shift
        
        # Orthogonal to tangent and radius:
        orthogonal = numpy.cross(radius, tangent)
        
        return position, tangent, radius, orthogonal
    
    
    def _euler2mat_(self, a, b, c, axes='sxyz'):
        a = numpy.deg2rad(a)
        b = numpy.deg2rad(b)
        c = numpy.deg2rad(c)
        
        return euler.euler2mat(a, b, c, axes)
    
    def _axangle2mat_(self, ax, a):
        a = numpy.deg2rad(a)
        
        return euler.axangle2mat(ax, a)
    
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
            ang_range: range of angles (default = 0..360)
            unit     : unit length (default = 'mm')
        '''
        # Parent init:
        basic.__init__(self, src2obj, det2obj, det_pixel, img_pixel, ang_range, unit)

        # Parameters:
        self.parameters.update({                    
                           'src_ort':0,   # source vertical, tangential and radial shifts
                           'src_tan':0,
                           
                           'det_ort':0,   # detector shifts and rotations
                           'det_tan':0,
                           'det_roll':0,
                           'det_pitch':0,
                           'det_yaw':0,
                           
                           'axs_tan':0,   # rotation axis shifts and rotations
                           'axs_roll':0,
                           'axs_pitch':0,                           
                           })    
            
    def detector_bounds(self, proj_shape):
        '''
        Get the boundaries of the detector in length units.
        '''   
        
        bounds = basic.detector_bounds(self, proj_shape)
        
        bounds[0] += self.parameters['det_ort']
        bounds[1] += self.parameters['det_tan']
    	
        return bounds    

    def volume_bounds(self, proj_shape):
        '''
        A very simplified version of volume bounds...
        '''
        # TODO: Compute this propoerly.... Dont trust the horizontal bounds!!!
        
        # Detector bounds:
        det_bounds = self.detector_bounds(proj_shape)
        
        vrt = det_bounds[0]
        hrz = det_bounds[1]
        
        # Demagnify detector bounds:
        fact = 1 / self.magnification
        vrt_bounds = (vrt * fact + self['src_ort'] * (1 - fact))
        hrz_bounds = (hrz * fact + self['src_tan'] * (1 - fact))
    
        max_x = max(abs(hrz_bounds - self['axs_tan']))
        
        hrz_bounds = [self['vol_tra'][2] - max_x, self['vol_tra'][2] + max_x]
        mag_bounds = [self['vol_tra'][1] - max_x, self['vol_tra'][1] + max_x]
                
        return numpy.array([vrt_bounds, mag_bounds, hrz_bounds]) 
    
    def _get_source_orbit_(self, angle_count = None, index = None):    
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
        src_vect, src_tan, src_rad, serc_orth = self._circular_orbit_(src2obj, angle_count, roll = axs_roll, pitch = axs_pitch, yaw = 0, 
                                                         origin = [0, 0, src_ort], tan_shift = src_tan - axs_tan, index = index)
        
        return src_vect
    
    def _get_detector_orbit_(self, angle_count = None, index = None):    
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
        det_pos, det_tan, det_rad, det_orth = self._circular_orbit_(det2obj, angle_count, roll = axs_roll, pitch = axs_pitch, yaw = 180, 
                                                         origin = [0, 0, det_ort], tan_shift = -det_tan + axs_tan, index = index)
        
        # Invert vectors to keep them alligned with the source vectors:
        det_tan, det_rad, det_orth = -det_tan, -det_rad, -det_orth
        
        # Apply detector rotations:    
        for ii in range(det_pos.shape[0]):
            
            T = self._axangle2mat_(det_rad[ii, :], det_roll)
            det_tan[ii, :] = T.dot(det_tan[ii, :])
            det_orth[ii, :] = T.dot(det_orth[ii, :])
        
            T = self._axangle2mat_(det_orth[ii, :], det_yaw)
            det_tan[ii, :] = T.dot(det_tan[ii, :])
            det_rad[ii, :] = T.dot(det_rad[ii, :])
            
            T = self._axangle2mat_(det_tan[ii, :], det_pitch)
            det_rad[ii, :] = T.dot(det_rad[ii, :])
            det_orth[ii, :] = T.dot(det_orth[ii, :])
        
        return det_pos, det_tan, det_rad, det_orth        
    