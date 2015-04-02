# plot difference from original and GIA

# Import python modules
import gdal
from gdalconst import *
import numpy
from scipy import ndimage
from scipy import stats
from matplotlib import pyplot as plt
import os, sys, shutil

# Define inputs
input_folder = '/Users/mc14909/Dropbox/Bristol/data/glacial-landscapes/'
output_folder = '/Users/mc14909/Dropbox/Bristol/scratch/glacial-landscapes/'
input_origDEM = input_folder + 'orig_DEM.tif'
input_giaDEM = input_folder + 'GIA_bedDEM_clipped2.tif'


# Open inputs
gorigDEM = gdal.Open(input_origDEM, GA_ReadOnly)
origDEM = gorigDEM.ReadAsArray(0, 0, gorigDEM.RasterXSize, gorigDEM.RasterYSize).astype(numpy.float)

ggiaDEM = gdal.Open(input_giaDEM, GA_ReadOnly)
giaDEM = ggiaDEM.ReadAsArray(0, 0, ggiaDEM.RasterXSize, ggiaDEM.RasterYSize).astype(numpy.float)

# Set nulls
origDEMnull = origDEM[0,0]
a = numpy.where(origDEM == origDEMnull)
origDEM[a] = numpy.NaN

giaDEMnull = giaDEM[0,0]
a = numpy.where(giaDEM == giaDEMnull)
giaDEM[a] = numpy.NaN

# Calculate difference image
diffDEM = giaDEM - origDEM

# Plots
fig_DEM = plt.figure(1)
fig_DEM.suptitle('GIA Difference Image', fontsize=12)
plt.imshow(diffDEM, vmin=-1000, vmax=1000)

plt.xlabel('Distance (km)', fontsize=10)
plt.ylabel('Distance (km)', fontsize=10)
cbar=plt.colorbar(extend='both')
cbar.set_label('Elevation (m)', fontsize=10)
plt.savefig(output_folder + 'GIA_diffDEM_plot.eps', dpi=1200)
