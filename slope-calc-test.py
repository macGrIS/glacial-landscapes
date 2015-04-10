# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 17:57:09 2015

https://github.com/MiXIL/calcSlopeDegrees/blob/master/calcSlopeDegrees.py

@author: mc14909
"""

import gdal
from gdalconst import *
import numpy as np
from scipy import ndimage
from scipy import stats
from math import sqrt
from matplotlib import pyplot as plt
import os, sys, shutil


def slopePython(inBlock, outBlock, inXSize, inYSize, zScale=1):

    """ Calculate slope using Python.
        If Numba is available will make use of autojit function
        to run at ~ 1/2 the speed of the Fortran module. 
        If not will fall back to pure Python - which will be slow!
    """
    for x in range(1,inBlock.shape[1]-1):
        for y in range(1, inBlock.shape[0]-1):
            # Get window size
            dx = 2 * inXSize[y,x]
            dy = 2 * inYSize[y,x]

            # Calculate difference in elevation
            dzx = (inBlock[y,x-1] - inBlock[y,x+1])*zScale
            dzy = (inBlock[y-1,x] - inBlock[y+1,x])*zScale

            # Find normal vector to the plane
            nx = -1 * dy * dzx
            ny = -1 * dx * dzy
            nz = dx * dy
    
            slopeRad = np.arccos(nz / np.sqrt(nx**2 + ny**2 + nz**2))
            slopeDeg = (180. / np.pi) * slopeRad
    
            outBlock[y,x] = slopeDeg
   
    return outBlock
    
input_folder = '/Users/mc14909/Dropbox/Bristol/data/glacial-landscapes/'
output_folder = '/Users/mc14909/Dropbox/Bristol/scratch/glacial-landscapes/'
inputDEM = input_folder + 'GIA_bedDEM_clipped2.tif'
inputSlope = input_folder + 'GDAL_slope2.tif'

gDEM = gdal.Open(inputDEM, GA_ReadOnly)
DEM = gDEM.ReadAsArray(0, 0, gDEM.RasterXSize, gDEM.RasterYSize).astype(np.float)
DEMnull = DEM[0,0] # input no data value for tiff
a = np.where(DEM == DEMnull) # finds all values in array DEM equal to nodata_value
DEM[a] = np.NaN # set nodata_value to NaN

gSlope = gdal.Open(inputSlope, GA_ReadOnly)
slope = gSlope.ReadAsArray(0, 0, gSlope.RasterXSize, gSlope.RasterYSize).astype(np.float)
print 'Slope sucessfully loaded into array...'
sNull = slope[0,0]
b =  np.where(slope == sNull) # finds all values in array equal to nodata value
slope[b] = np.NaN

# Start slope setup
inBlock = DEM
outBlock = np.zeros(DEM.shape)
inXSize = np.ones(DEM.shape)
inYSize = np.ones(DEM.shape)

outBlock = slopePython(inBlock, outBlock, inXSize, inYSize, zScale=1)

outBlock_test = outBlock[1:3000,1:2500]
slope_test = slope[1:3000,1:2500]

test = np.array_equal(slope_test,outBlock_test)
print test

fig_DEM = plt.figure(1)
fig_DEM.suptitle('Greenland bedrock GIA DEM', fontsize=12)
plt.imshow(DEM, cmap='terrain', vmax=2000, vmin=-800) # vmax is a way of stretching colour map
plt.xlabel('Distance (km)', fontsize=10)
plt.ylabel('Distance (km)', fontsize=10)
cbar=plt.colorbar(extend='both')
cbar.set_label('Elevation (m)', fontsize=10)
plt.savefig(output_folder + 'DEM_plot.eps', dpi=1200)
print 'DEM plotted successfully -- ' + output_folder + 'DEM_plot.eps'

# Slope
fig_slope = plt.figure(2)
fig_slope.suptitle('Slope of Greenland bedrock (GIA) DEM', fontsize=12)
plt.imshow(slope, cmap='jet_r') # _r is a way to reverse colour map
plt.xlabel('Distance (km)', fontsize=10)
plt.ylabel('Distance (km)', fontsize=10)
cbar=plt.colorbar(extend='neither')
cbar.set_label('slope', fontsize=10)
plt.savefig(output_folder + 'slope_plot.eps', dpi=1200)
print 'Slope plotted successfully -- ' + output_folder + 'slope_plot.eps'

# Slope
fig_slope = plt.figure(3)
fig_slope.suptitle('Slope TEST', fontsize=12)
plt.imshow(outBlock, cmap='jet_r') # _r is a way to reverse colour map
plt.xlabel('Distance (km)', fontsize=10)
plt.ylabel('Distance (km)', fontsize=10)
cbar=plt.colorbar(extend='neither')
cbar.set_label('slope', fontsize=10)
plt.savefig(output_folder + 'TEST_slope.eps', dpi=1200)
print 'Slope plotted successfully -- ' + output_folder + 'slope_plot.eps'
