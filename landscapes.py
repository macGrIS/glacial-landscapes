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
import scipy
from scipy import ndimage
import matplotlib
from matplotlib import pyplot as plt
import os, sys, shutil

### Functions
# scipy.ndimage used to calculate ranges (and min and max) of values (array) for different size grids (factor)
def grid_range(array, factor):
    assert isinstance(factor, int), type(factor)
    sx, sy = array.shape
    X, Y = numpy.ogrid[0:sx, 0:sy]
    regions = sy/factor * (X/factor) + (Y/factor)
    block_max = ndimage.maximum(array, labels=regions, index=numpy.arange(regions.max() + 1))
    block_max.shape = (sx/factor, sy/factor)
    block_min = ndimage.minimum(array, labels=regions, index=numpy.arange(regions.max() + 1))
    block_min.shape = (sx/factor, sy/factor)
    block_range = block_max - block_min
    return block_max, block_min, block_range, regions;

### Define input
input_folder = '/Users/mc14909/Dropbox/Bristol/data/glacial-landscapes/'
output_folder = '/Users/mc14909/Dropbox/Bristol/scratch/glacial-landscapes/'
inputDEM = input_folder + 'GIA_bedDEM_clipped2.tif'
inputSlope = input_folder + 'GDAL_slope2.tif' # must be a way to calculate this (see below), until then, using gdaldem slope output

### Define parameters and metrics
# size of grid/ fishnet (in km)
factor = 100
print 'Running with a gridsize of: ' + str(factor)

## Decision Tree
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

# Check user for No Data value??
# DEM no data to NaN
DEMnull = DEM[0,0] # input no data value for tiff
a = numpy.where(DEM == DEMnull) # finds all values in array DEM equal to nodata_value
DEM[a] = numpy.NaN # set nodata_value to NaN
print 'Null values set.'

### Calculate metrics
# Input subset?


# Calculate slope
""" 
Now, I'm not a mathematician, so I'm not sure how this works -- but this method 
of calculating slope is not correct (yields a roughly inverse, but still different
result than that of gdaldem slope, and ArcGIS slope...)

Until I work this out, we use GDAL output...

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
b = numpy.where(slope == sNull) # finds all values in array equal to nodata value
slope[b] = numpy.NaN # set nodata value to NaN
print 'Null values set.'


## Calculate peak analysis
# Identify peaks
"""
    need to open DEM, and identify high points/ peaks within array of 1000, 1500, 2000, 2500 and 3000 metres with a minimum drop surrounding peak as 250 m
http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

"""
# Set up blank grid
peaky, peakx = DEM[0:3000,0:2500].shape
p = numpy.zeros([peaky/factor,peakx/factor]) # new grid of factor size size -- to fill with peak density data

# convolution? moving window size -- then find peaks?

"""
use window to go through array DEM

fill 6 arrays

one -- if cell is >= 1000 and difference between all surrounding cells is greater than 250 (all peaks)
2 -- if cell is >= 1000 and < 1500 and diff between etc.
3 -- if cell is >= 1500 and < 2000 and " "
4 -- if cell is >= 2000 and < 2500 and " "
5 -- if cell is >= 2500 and < 3000 and " "
6 -- if cell is >= 3000 and " "
"""

# Peak density
"""
density of peaks per fishnet

point density
"""

## Calculate elevation range
DEM_subset = DEM[0:3000, 0:2500]
elev_max, elev_min, elev_range, regions = grid_range(DEM_subset, factor)

numpy.savetxt(output_folder + 'elev_range.txt', elev_range)
print 'Elevation range successfully calculated.'

## Calculate slope range
slope_subset = slope[0:3000,0:2500]
slope_max, slope_min, slope_range, regions = grid_range(slope_subset, factor)

numpy.savetxt(output_folder + 'slope_range.txt', slope_range)
print 'Slope range successfully calculated.'

# Need to classify these... put into binary grid?

## Calculate hypsometry (elevation over area)

# Monumentally slow peice of code ... 
# NOT CORRECT
regions_list = regions.ravel() # turns grid label into 1d array
DEM_list = DEM_subset.ravel() # turns DEM into 1d array (conforming to regions_list)
hypso_a = numpy.concatenate(([regions_list], [DEM_list]),axis=0)
last_box = numpy.max(hypso_a[0,]) # finds the last label for the grid
hypso_b = numpy.swapaxes(hypso_a, 0, 1)

for i in range(0,4): # 0,last_box
    A = numpy.where(hypso_b[:,0] == float(i))
    hypso_c = hypso_b[A,1]
    hypso_d = numpy.swapaxes(hypso_c, 0, 1)
    # get shape, if shape is < it should (due to NaNs) then don't run??
    if numpy.shape(numpy.isnan(hypso_d[:,0])) > 1000: # 1000 needs a function
        print 'Too many NaN values for cell: ' + str(i) + ', skipping...'
        continue
    plt.hist(hypso_d[~numpy.isnan(hypso_d)], bins=100) # get rid of remaining nans (shouldn't be more than X% of total)
    plt.show()



# now to plot per unique values?!?! -- hsplit?!

# now plot histograms by the first row?

""" slice array iteratively -- extracting values into histograms """

#for i in DEM_subset:
#    subset = DEM_subset[i*factor:i*factor+factor]


    

# ndimage.find_objects(DEM_subset) ???

# Skewness test -- threshold

# Bimodal test -- threshold


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
# Plot Peak Density

# Plot Elevation Range
fig_DEM = plt.figure(4)
fig_DEM.suptitle('elev_range', fontsize=12)
plt.imshow(elev_range, interpolation='nearest', extent=[0,2500,0,3000]) # interpolation 'nearest' stops blurry figures!

plt.xlabel('Distance (km)', fontsize=10)
plt.ylabel('Distance (km)', fontsize=10)
cbar=plt.colorbar(extend='neither')
cbar.set_label('Elevation range (m)', fontsize=10)
plt.savefig(output_folder + 'DEM_elev_range.eps', dpi=1200)

print 'Elevation range plotted successfully -- ' + output_folder + 'DEM_elev_range.eps'

# Plot Slope Range
fig_DEM = plt.figure(5)
fig_DEM.suptitle('slope_range', fontsize=12)
plt.imshow(slope_range, interpolation='nearest', extent=[0,2500,0,3000])

plt.xlabel('Distance (km)', fontsize=10)
plt.ylabel('Distance (km)', fontsize=10)
cbar=plt.colorbar(extend='neither')
cbar.set_label('Slope range (deg)', fontsize=10)
plt.savefig(output_folder + 'DEM_slope_range.eps', dpi=1200)

print 'Slope range plotted successfully -- ' + output_folder + 'DEM_slope_range.eps'

# Plot Classified Grid

#X,Y = numpy.meshgrid(2501,3001)


# Create fishnet/ subset grid


# need to set up geolocation/ geotransform

# - create a fishnet of variable size (as in set the size as a variable -- look into fuzzy boxes later)
# - produce a result of elevation range, and slope range (as per fishnet)
# - produce a fishnet with smaller arrays for elevation values per fishnet box
# - produce histograms from fishnet (hypsometry)
# 		-	work out skewness, and also smooth to work out bimodal??

# plot results!