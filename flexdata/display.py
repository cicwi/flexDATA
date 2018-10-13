#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 2018
@author: kostenko

Few simple reoutines for displaying 3D data (memmap-compatible).

"""

''' * Imports * '''

import numpy
import matplotlib.pyplot as plt
from matplotlib import ticker
from . import array 

''' * Methods * '''
    
def plot(x, y = None, semilogy = False, title = None, legend = None):
    
    if y is None:
        y = x
        x = numpy.arange(numpy.size(x))
    
    x = numpy.squeeze(x)
    y = numpy.squeeze(y)
    
    plt.figure()
    if semilogy:
        plt.semilogy(x, y)
    else:
        plt.plot(x, y)
    
    if title:
        plt.title(title)
    
    if legend:
        plt.legend(legend)
        
    plt.show()    

def display_slice(data, index = None, dim = 0, bounds = None, title = None, cmap = 'gray', file = None):
    
    # Just in case squeeze:
    data = numpy.squeeze(data)
    
    # If the image is 2D:
    if data.ndim == 2:
        img = data
        
    else:
        if index is None:
            index = data.shape[dim] // 2
    
        sl = array.anyslice(data, index, dim)

        img = numpy.squeeze(data[sl])
        
        # There is a bug in plt. It doesn't like float16
        if img.dtype == 'float16': img = numpy.float32(img)
        
    fig = plt.figure()
    
    if bounds:
        imsh = plt.imshow(img, vmin = bounds[0], vmax = bounds[1], cmap = cmap)
    else:
        imsh = plt.imshow(img, cmap = cmap)
        
    #plt.colorbar()
    cbar = fig.colorbar(imsh, ticks = ticker.MaxNLocator(nbins=6))
    cbar.ax.tick_params(labelsize=15) 
    
    plt.axis('off')
    
    if title:
        plt.title(title)
        
    plt.show()  
    
    if file:
        plt.savefig(file, dpi=300, bbox_inches='tight')
    
def display_mesh(stl_mesh):
    """
    Display an stl mesh. Use flexCompute.generate_stl(volume) to generate mesh.
    """    
    from mpl_toolkits import mplot3d
        
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)

    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(stl_mesh.vectors))
    
    # Auto scale to the mesh size
    scale = stl_mesh.points.flatten(-1)
    axes.auto_scale_xyz(scale, scale, scale)
    # Show the plot to the screen
    plt.show()


def display_projection(data, dim = 1, bounds = None, title = None, cmap = 'gray', file = None):
    
    img = data.sum(dim)
    
    # There is a bug in plt. It doesn't like float16
    img = numpy.float32(img)
    
    plt.figure()
    
    if bounds:
        plt.imshow(img, vmin = bounds[0], vmax = bounds[1], cmap = cmap)
    else:
        plt.imshow(img, cmap = cmap)
        
    plt.colorbar()
    plt.axis('off')
    
    if title:
        plt.title(title)
    
    plt.show()
    
    if file:
        plt.savefig(file, dpi=300, bbox_inches='tight')
    
def display_max_projection(data, dim = 0, bounds = None, title = None, cmap = 'gray', file = None):
    
    img = data.max(dim)
    
    # There is a bug in plt. It doesn't like float16
    img = numpy.float32(img)
    
    plt.figure()
    
    if bounds:
        plt.imshow(img, vmin = bounds[0], vmax = bounds[1], cmap = cmap)
    else:
        plt.imshow(img, cmap = cmap)
    
    plt.colorbar()
    plt.axis('off')
    
    if title:
        plt.title(title)     
        
    plt.show()
    
    if file:
        plt.savefig(file, dpi=300, bbox_inches='tight')
        
def display_min_projection(data, dim = 0, title = None, cmap = 'gray', file = None):
    
    img = data.min(dim)
    
    # There is a bug in plt. It doesn't like float16
    img = numpy.float32(img)
    
    plt.figure()
    
    plt.imshow(img, cmap = cmap)
    plt.colorbar()
    plt.axis('off')
    
    if title:
        plt.title(title)         
        
    plt.show()
    
    if file:
        plt.savefig(file, dpi=300, bbox_inches='tight')