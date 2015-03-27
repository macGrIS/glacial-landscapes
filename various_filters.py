# filters

### Header
scriptname = 'Various filters'
print 'Running script: ' + scriptname + '.'

# Import python modules
import gdal
from gdalconst import *
import numpy
from scipy import ndimage
from scipy import stats
from matplotlib import pyplot as plt
import os, sys, shutil

### Define input
input_folder = '/Users/mc14909/Dropbox/Bristol/data/glacial-landscapes/'
output_folder = '/Users/mc14909/Dropbox/Bristol/scratch/glacial-landscapes/'
inputDEM = input_folder + 'GIA_bedDEM_clipped2.tif'
inputSlope = input_folder + 'GDAL_slope2.tif' # must be a way to calculate this (see below), until then, using gdaldem slope output

### Define parameters and metrics
# size of grid/ fishnet (in km -- for 1km posting data)
factor = 50
print 'Grid cell size = ' + str(factor) + ' km.' # units depend on pixel 'size'

### Call input DEM
## Open DEM
gDEM = gdal.Open(inputDEM, GA_ReadOnly)
DEM = gDEM.ReadAsArray(0, 0, gDEM.RasterXSize, gDEM.RasterYSize).astype(numpy.float)
print 'DEM sucessfully loaded into array...'

# DEM no data to NaN
DEMnull = DEM[0,0] # input no data value for tiff
a = numpy.where(DEM == DEMnull) # finds all values in array DEM equal to nodata_value
DEM[a] = numpy.NaN # set nodata_value to NaN
print 'Null values set.'

## Open slope
gSlope = gdal.Open(inputSlope, GA_ReadOnly)
print 'Slope loaded sucessfully, checking file type...'
slope = gSlope.ReadAsArray(0, 0, gSlope.RasterXSize, gSlope.RasterYSize).astype(numpy.float)
print 'Slope sucessfully loaded into array...'

# Slope no data to NaN
sNull = slope[0,0]
b = numpy.where(slope == sNull) # finds all values in array equal to nodata value
slope[b] = numpy.NaN # set nodata value to NaN
print 'Null values set.'

### Filters

## Gaussian
gaussian = ndimage.filters.gaussian_filter(DEM, factor/2, order=0,mode='reflect')


crevasse_array = DEM - gaussian

### Plots
# Gaussian
fig_DEM = plt.figure(1)
fig_DEM.suptitle('Greenland bedrock GIA DEM -- Gaussian Filter', fontsize=12)
plt.imshow(gaussian)
plt.xlabel('Distance (km)', fontsize=10)
plt.ylabel('Distance (km)', fontsize=10)
cbar=plt.colorbar(extend='both')
plt.savefig(output_folder + 'gaussian_plot.eps', dpi=1200)

# Plot slope
fig_slope = plt.figure(2)
fig_slope.suptitle('Difference', fontsize=12)
plt.imshow(crevasse_array) # _r is a way to reverse colour map

plt.xlabel('Distance (km)', fontsize=10)
plt.ylabel('Distance (km)', fontsize=10)
cbar=plt.colorbar(extend='neither')
plt.savefig(output_folder + 'difference_plot.eps', dpi=1200)

#
## Plot Elevation Range
#fig_DEM = plt.figure(4)
#fig_DEM.suptitle('elev_range', fontsize=12)
#plt.imshow(elev_range, interpolation='nearest', extent=[0,2500,0,3000]) # interpolation 'nearest' stops blurry figures!
#
#plt.xlabel('Distance (km)', fontsize=10)
#plt.ylabel('Distance (km)', fontsize=10)
#cbar=plt.colorbar(extend='neither')
#cbar.set_label('Elevation range (m)', fontsize=10)
#plt.savefig(output_folder + 'DEM_elev_range' + str(factor) + '.eps', dpi=1200)
#
#print 'Elevation range plotted successfully -- ' + output_folder + 'DEM_elev_range.eps'
#
## Plot Slope Range
#fig_DEM = plt.figure(5)
#fig_DEM.suptitle('slope_range', fontsize=12)
#plt.imshow(slope_range, interpolation='nearest', extent=[0,2500,0,3000])
#
#plt.xlabel('Distance (km)', fontsize=10)
#plt.ylabel('Distance (km)', fontsize=10)
#cbar=plt.colorbar(extend='neither')
#cbar.set_label('Slope range (deg)', fontsize=10)
#plt.savefig(output_folder + 'DEM_slope_range' + str(factor) + '.eps', dpi=1200)
#
#print 'Slope range plotted successfully -- ' + output_folder + 'DEM_slope_range.eps'