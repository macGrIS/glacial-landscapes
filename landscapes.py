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

### Define input
input_folder = '/Users/mc14909/Dropbox/Bristol/data/glacial-landscapes/'
output_folder = '/Users/mc14909/Dropbox/Bristol/scratch/glacial-landscapes/'
inputDEM = input_folder + 'GIA_bedDEM_clipped.tif'
inputSlope = input_folder + 'GDAL_slope.tif' # must be a way to calculate this (see below), until then, using gdaldem slope output

### Define parameters and metrics
# size of fishnet
grid_size = 100
# elevation range threshold lower

# elevation range threshold upper

# peak density threshold

# slope threshold


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

# Check user for No Data value?
# DEM no data to NaN
DEMnull = DEM[0,0] # input no data value for tiff
a = numpy.where(DEM == DEMnull) # finds all values in array DEM equal to nodata_value
DEM[a] = numpy.NaN # set nodata_value to NaN



### Calculate metrics

## Calculate slope
""" 
Now I'm not a mathematician, so I'm not sure how this works -- but this method 
of calculating slope is not correct (yeilds a roughly inverse, but still different
result than that of gdaldem slope, and ArcGIS slope...)

Until I work this out, we use GDAL output

x, y = numpy.gradient(DEM, X) # calculate gradient with sample distance X (define)

slope_test = numpy.pi/2. - numpy.arctan(numpy.sqrt(x*x + y*y))
for i in numpy.nditer(slope_test, op_flags=['readwrite']):
    i[...] = numpy.degrees(i)


#rad2deg = 180.0 / math.pi
#slope_test2 = 90.0 - arctan(sqrt(x*x + y*y)) * rad2deg
#
#c = numpy.where(numpy.isnan(slope_test))
#slope_test[c] = -9999
#numpy.savetxt(output_folder + 'python_slope_test.txt', slope_test)
#
#d = numpy.where(numpy.isnan(slope_test2))
#slope_test2[c] = -9999
#numpy.savetxt(output_folder + 'python_slope_test2.txt', slope_test2)
"""

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

""" # Plot hypsometry ??? """

# Skewness test

""" skewness threshold? """

# Bimodal test

""" bimodal threshold """

### Plot inputs
# Plot DEM
fig_DEM = plt.figure(1)
fig_DEM.suptitle('Greenland bedrock GIA DEM', fontsize=12)
plt.imshow(DEM, cmap='terrain', vmax=2000, vmin=-800) # vmax is a way of stretching colour map

plt.xlabel('Distance (km)', fontsize=10)
plt.ylabel('Distance (km)', fontsize=10)
cbar=plt.colorbar(extend='both')
cbar.set_label('Elevation (m)', fontsize=10)
plt.savefig(output_folder + 'DEM_plot.eps', dpi=1200)

print 'DEM plotted successfully -- ' + output_folder + 'DEM_plot.eps'

# Plot slope
fig_slope = plt.figure(2)
fig_slope.suptitle('Slope of Greenland bedrock (GIA) DEM', fontsize=12)
plt.imshow(slope, cmap='jet_r') # _r is a way to reverse colour map

plt.xlabel('Distance (km)', fontsize=10)
plt.ylabel('Distance (km)', fontsize=10)
cbar=plt.colorbar(extend='neither')
cbar.set_label('slope', fontsize=10)
plt.savefig(output_folder + 'slope_plot.eps', dpi=1200)

print 'Slope plotted successfully -- ' + output_folder + 'slope_plot.eps'

### Plot outputs
# Plot overall grid


#X,Y = numpy.meshgrid(2501,3001)


# Create fishnet/ subset grid


# need to set up geolocation/ geotransform

# - create a fishnet of variable size (as in set the size as a variable -- look into fuzzy boxes later)
# - produce a result of elevation range, and slope range (as per fishnet)
# - produce a fishnet with smaller arrays for elevation values per fishnet box
# - produce histograms from fishnet (hypsometry)
# 		-	work out skewness, and also smooth to work out bimodal??

# plot results!

