#-----------------------------------------------------------------------------------------
#	4th March, 2015; 18:12 GMT
#	Script by Michael A. Cooper (t: @macooperr; git: @macGrIS)
#	to produce classification maps of 'landscapes of glacial erosion' (as per Jamieson, et
#	al., 2014; Sugden and John, 1975) from GeoTIFF DEMs.
#-----------------------------------------------------------------------------------------

### Header
scriptname = 'Landscapes of Glacial Erosion -- Classifier'
print 'Running script: ' + scriptname + '.'

# Import python modules
import gdal
from gdalconst import *
import numpy
import matplotlib
from matplotlib import pyplot as plt
import os, sys, shutil

### Define variables
input_folder = '/Users/mc14909/Dropbox/Bristol/data/glacial-landscapes/'
output_folder = '/Users/mc14909/Dropbox/Bristol/scratch/glacial-landscapes/'
inputDEM = input_folder + 'GIA_bedDEM_clipped_nodata.tif'
inputSlope = input_folder + 'GIA_bedDEM_slope.tif' # must be a way to calculate this

grid_size = 100 # size of fishnet

### Call input DEM
# Open DEM
gDEM = gdal.Open(inputDEM, GA_ReadOnly)
print 'DEM loaded sucessfully, checking file type...'

# Check input DEM is GeoTIFF
driver = gDEM.GetDriver().LongName
if driver == 'GeoTIFF': # continues if GeoTIFF
	print 'File type supported, continuing.'
else:                   # aborts if not GeoTIFF
	sys.exit('File type not supported, aborting.')

# Read DEM into numpy array
DEM = gDEM.ReadAsArray(0, 0, gDEM.RasterXSize, gDEM.RasterYSize).astype(numpy.float)
print 'DEM sucessfully loaded into array.'

# Check user for No Data?
# DEM no data to NaN
DEMnull = -9999
a = numpy.where(DEM == DEMnull) # finds all values in array DEM equal to nodata_value
DEM[a] = numpy.NaN # set nodata_value to NaN


### Calculate metrics
## Calculate slope -- must be a way to calculate slope from DEM, for now open from .tiff created in Arc
# Open slope
gSlope = gdal.Open(inputSlope, GA_ReadOnly)
print 'Slope loaded sucessfully, checking file type...'

# Check input slope is GeoTIFF
driver = gSlope.GetDriver().LongName
if driver == 'GeoTIFF': # continues if GeoTIFF
    print 'File type supported, continuing.'
else:                   # aborts if not GeoTIFF
    sys.exit('File type not supported, aborting.')

# Read slope into numpy array
slope = gSlope.ReadAsArray(0, 0, gSlope.RasterXSize, gSlope.RasterYSize).astype(numpy.float)
print 'Slope sucessfully loaded into array.'

# Slope no data to NaN
sNull = slope[0,0]
b = numpy.where(slope == sNull) # finds all values in array DEM equal to nodata_value
slope[b] = numpy.NaN # set nodata_value to NaN

## Calculate peak analysis
# Identify peaks
"""
    need to open DEM, and identify high points/ peaks within array of 1000, 1500, 2000, 2500 and 3000 metres with a minimum drop surrounding peak as 250 m
http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

"""

# Peak density
"""
    density of peaks per fishnet
"""

## Calculate elevation range


## Calculate slope range


## Calculate hypsometry (elevation over area)

### Plot outputs
# Plot DEM
DEMplot = plt.imshow(DEM, cmap='bone')
plt.savefig(output_folder + 'DEM_plot.png')
print 'DEM plotted successfully -- ' + output_folder + 'DEM_plot.png'

# Plot slope
Splot = plt.imshow(slope, cmap='bone')
plt.savefig(output_folder + 'Slope_plot.png')
print 'Slope plotted successfully -- ' + output_folder + 'Slope_plot.png'



#X,Y = numpy.meshgrid(2501,3001)


# Create fishnet/ subset grid


# need to set up geolocation/ geotransform

# - create a fishnet of variable size (as in set the size as a variable -- look into fuzzy boxes later)
# - produce a result of elevation range, and slope range (as per fishnet)
# - produce a fishnet with smaller arrays for elevation values per fishnet box
# - produce histograms from fishnet (hypsometry)
# 		-	work out skewness, and also smooth to work out bimodal??

# plot results!

