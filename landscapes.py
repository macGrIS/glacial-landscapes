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
from scipy import ndimage
from scipy import stats
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
# size of grid/ fishnet (in km -- for 1km posting data)
factor = 50

print 'Grid cell size = ' + str(factor) + ' km.' # units depend on pixel 'size'

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
print 'DEM sucessfully loaded into array...'

# Check user for No Data value??
# DEM no data to NaN
DEMnull = DEM[0,0] # input no data value for tiff
a = numpy.where(DEM == DEMnull) # finds all values in array DEM equal to nodata_value
DEM[a] = numpy.NaN # set nodata_value to NaN
print 'Null values set.'

### Calculate metrics
# Input subset?


# Calculate slope
print 'Calculating slope...'
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
print 'Slope sucessfully loaded into array...'

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
print 'Calculating elevation range per grid cell...'
DEM_subset = DEM[0:3000, 0:2500]
elev_max, elev_min, elev_range, regions = grid_range(DEM_subset, factor)
# numpy.savetxt(output_folder + 'elev_range.txt', elev_range)
print 'Done.'

elev_range_list = elev_range.ravel()
#elev_test = 
#elev_test[elev_range_list>elev_thres_1] = 1

## Calculate slope range
slope_subset = slope[0:3000,0:2500]
slope_max, slope_min, slope_range, regions = grid_range(slope_subset, factor)
print 'Calculating elevation range per grid cell...'
# numpy.savetxt(output_folder + 'slope_range.txt', slope_range)
print 'Done.'

slope_range_list = slope_range.ravel()

"""  Need to classify these... put into binary, then grid (based upon thresholds above)? """

## Calculate hypsometry (elevation over area) -- chuck this into a defined function?
# Monumentally inefficient loop (probably...)
print 'Calculating hypsometry per grid cell...'
regions_list = regions.ravel() # turns grid label into 1d array
DEM_list = DEM_subset.ravel() # turns DEM into 1d array (conforming to regions_list)
hypso_a = numpy.concatenate(([regions_list], [DEM_list]),axis=0)
last_box = 1+numpy.max(hypso_a[0,]) # finds the last label for the grid
hypso_b = numpy.swapaxes(hypso_a, 0, 1)
skewness_test = numpy.zeros([last_box, 1])
bimodal_test = numpy.zeros([last_box, 1])

for i in range(0,int(last_box)): # 0,last_box
    A = numpy.where(hypso_b[:,0] == float(i))
    hypso_c = hypso_b[A,1]
    hypso_d = numpy.swapaxes(hypso_c, 0, 1)
    # Weed out null data/ NaNs
    null_data = numpy.isnan(hypso_d[:,0]).any() # too perscriptive? 
    # null_data = numpy.isnan(hypso_d[:,0]).all() # too liberal? maybe use 10%?? (not sure how, also arbitrary)
    if null_data == True: 
        print 'Too many NaN values for cell: ' + str(i) + ', skipping...'
        continue
    else:
        fig_hist = plt.figure(1)
        n, bins, patches = plt.hist(hypso_d[~numpy.isnan(hypso_d)], bins=100) # Histo plot, get rid of remaining NaNs
        #plt.show() # can output plots, but slow...!!
        
    ## Skewness test
    skew = stats.skew(hypso_d[~numpy.isnan(hypso_d)]) # perform skewness measure on all but NaN values
    if skew > 0.1: # Any grid with a skew of greater than 0.1 is classed as positively skewed (change this to threshold up top)
        skewness_test[i,0] = 1
    else:
        skewness_test[i,0] = 0
        
    ## Bimodal test -- are there other ways to do this?
    # Thresholds    
    x_threshold = 0.4 # X-axis (Distance/ Location) threshold -- a percentage of the range of the data, bins further away from this (around peak bin) can be considered separate peaks
    y_threshold = 0.2 # Y-axis (Peak) threshold -- a percentage of the count in the peak bin, bins with more members can be considered peaks
    
    bincentres = 0.5*(bins[1:]+bins[:-1]) # find bincenters (as bins produces a 101 length array)
    histo_data = numpy.concatenate(([n],[bincentres])) # joins n (number in bin) with bincentres (bin placement)
    histo_data = numpy.swapaxes(histo_data, 0, 1)
    
    # Rank histo data -- to find peak
    bins_ranked = numpy.concatenate(([numpy.sort(n)],[numpy.argsort(n, axis=0)]))
    bins_ranked = numpy.swapaxes(bins_ranked, 0, 1)
    sort = numpy.flipud(bins_ranked)
    
    peak_location = sort[0,1] # the rank of the bin location -- where the bin (out of 100 along x) was
    peak_count = sort[0,0] # the count of the largest bin count -- how many values were that bin
    
    # Calculate distance from the largest bin
    distance = numpy.sqrt((histo_data[:,1]-histo_data[peak_location,1])**2) 
    
    # Calculate threshold above which data can be considered a peak
    peak_threshold = y_threshold*peak_count
    
    # Calculate threshold distance over which data can be considered a separate peak
    data_range = (numpy.max(histo_data[:,1]-numpy.min(histo_data[:,1])))
    distance_threshold = x_threshold*data_range

    # Test   
    test = numpy.zeros([100, 1])
    test[histo_data[:,0]>peak_threshold]=1
    test[distance<distance_threshold]=0

    # Result
    bimodal_test[i,0]=numpy.sum(test)

bimodal_test[bimodal_test>0]=1 # make bimodal_test binary

""" now need to output to grid? opposite of ravel? """

### Plot inputs
# Plot DEM
fig_DEM = plt.figure(2)
fig_DEM.suptitle('Greenland bedrock GIA DEM', fontsize=12)
plt.imshow(DEM, cmap='terrain', vmax=2000, vmin=-800) # vmax is a way of stretching colour map

plt.xlabel('Distance (km)', fontsize=10)
plt.ylabel('Distance (km)', fontsize=10)
cbar=plt.colorbar(extend='both')
cbar.set_label('Elevation (m)', fontsize=10)
plt.savefig(output_folder + 'DEM_plot.eps', dpi=1200)

print 'DEM plotted successfully -- ' + output_folder + 'DEM_plot.eps'

# Plot slope
fig_slope = plt.figure(3)
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
plt.savefig(output_folder + 'DEM_elev_range' + str(factor) + '.eps', dpi=1200)

print 'Elevation range plotted successfully -- ' + output_folder + 'DEM_elev_range.eps'

# Plot Slope Range
fig_DEM = plt.figure(5)
fig_DEM.suptitle('slope_range', fontsize=12)
plt.imshow(slope_range, interpolation='nearest', extent=[0,2500,0,3000])

plt.xlabel('Distance (km)', fontsize=10)
plt.ylabel('Distance (km)', fontsize=10)
cbar=plt.colorbar(extend='neither')
cbar.set_label('Slope range (deg)', fontsize=10)
plt.savefig(output_folder + 'DEM_slope_range' + str(factor) + '.eps', dpi=1200)

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