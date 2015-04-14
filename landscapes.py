# -*- coding: utf-8 -*-
"""
	4th March, 2015; 18:12 GMT
	Script by Michael A. Cooper (t: @macooperr; git: @macGrIS)
	to produce classification maps of 'landscapes of glacial erosion' (as per Jamieson, et
	al., 2014; Sugden and John, 1975) from GeoTIFF DEMs.
"""

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
    block_sum = ndimage.sum(array, labels=regions, index=numpy.arange(regions.max() + 1))
    block_sum.shape = (sx/factor, sy/factor)
    return block_max, block_min, block_range, block_sum, regions;

### Ask user for deets...??
### Define input
input_folder = '/Users/mc14909/Dropbox/Bristol/data/glacial-landscapes/'
output_folder = '/Users/mc14909/Dropbox/Bristol/scratch/glacial-landscapes/'
inputDEM = input_folder + 'GIA_bedDEM_clipped2.tif'
inputSlope = input_folder + 'GDAL_slope2.tif' # must be a way to calculate this (see below), until then, using gdaldem slope output

### Define parameters and metrics
## size of grid/ fishnet (in km -- for 1km posting data) -- LOOK INTO FUZZY BOXES LATER
factor = 25 # doesn't seem to work for 150, or 200km grid sizes (why?)
print 'Grid cell size = ' + str(factor) + ' km.' # units depend on pixel 'size'

## Input subsets (specific catchments?) - "do you wish to subset the data?" (perhaps at end...?)
#local_factor = 50 # factor of subset

## Decision Tree (based upon Jamieson, et al. 2014) -- Change for 'sensitivity analysis' -- ask for custom, or standard.
# Elevation range threshold lower
#tree_elev_lower_thres = 1000 # orig < 1000 m
# Elevation range threshold upper
# Peak density threshold
#tree_elev_upper_thres = 2000 # orig >= 1000 and < 2000 m
#tree_pd_thres = 20 # orig <= 25
# Slope threshold
#tree_slope_thres = 45 # orig <= 5

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

## Set up blank grid based upon factor
peaky, peakx = DEM[0:3000,0:2500].shape
p = numpy.zeros([peaky/factor,peakx/factor]) # new grid of factor size size -- to fill with peak density data

### Calculate metrics
## Calculate slope -- need to add in this (possibly solved through convolution)
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
# Open peak detection results from Landserf -- value of 5 depicts summit (to use for peak density)
# Currently only using peaks1000 (as it displays ALL peaks over 1000 m in elevation) -- may require others later
input_peaks1000 = input_folder + 'peak_anal/peaks_1000.tif'
#input_peaks1500 = input_folder + 'peak_anal/peaks_1500.tif'
#input_peaks2000 = input_folder + 'peak_anal/peaks_2000.tif'
#input_peaks2500 = input_folder + 'peak_anal/peaks_2500.tif'
#input_peaks3000 = input_folder + 'peak_anal/peaks_3000.tif'

gPeaks1000 = gdal.Open(input_peaks1000, GA_ReadOnly)
peaks1000 = gPeaks1000.ReadAsArray(0, 0, gPeaks1000.RasterXSize, gPeaks1000.RasterYSize).astype(numpy.float)
#gPeaks1500 = gdal.Open(input_peaks2500, GA_ReadOnly)
#peaks1500 = gPeaks1500.ReadAsArray(0, 0, gPeaks1500.RasterXSize, gPeaks1500.RasterYSize).astype(numpy.float)
#gPeaks2000 = gdal.Open(input_peaks2000, GA_ReadOnly)
#peaks2000 = gPeaks2000.ReadAsArray(0, 0, gPeaks2000.RasterXSize, gPeaks2000.RasterYSize).astype(numpy.float)
#gPeaks2500 = gdal.Open(input_peaks2500, GA_ReadOnly)
#peaks2500 = gPeaks2500.ReadAsArray(0, 0, gPeaks2500.RasterXSize, gPeaks2500.RasterYSize).astype(numpy.float)
#gPeaks3000 = gdal.Open(input_peaks3000, GA_ReadOnly)
#peaks3000 = gPeaks3000.ReadAsArray(0, 0, gPeaks3000.RasterXSize, gPeaks3000.RasterYSize).astype(numpy.float)

peaks1000[numpy.logical_or(peaks1000 > 5., peaks1000 < 5.)] = 0.
peaks1000[peaks1000 == 5.] = 1.
#peaks1500[numpy.logical_or(peaks1500 > 5., peaks1500 < 5.)] = 0.
#peaks1500[peaks1500 == 5.] = 1.
#peaks2000[numpy.logical_or(peaks2000 > 5., peaks2000 < 5.)] = 0.
#peaks2000[peaks2000 == 5.] = 1.
#peaks2500[numpy.logical_or(peaks2500 > 5., peaks2500 < 5.)] = 0.
#peaks2500[peaks2500 == 5.] = 1.
#peaks3000[numpy.logical_or(peaks3000 > 5., peaks3000 < 5.)] = 0.
#peaks3000[peaks3000 == 5.] = 1.


# Identify peaks
"""
need to open DEM, and identify high points/ peaks within array of 1000, 1500,
2000, 2500 and 3000 metres with a minimum drop surrounding peak as 250 m
http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

# convolution? moving window size -- then find peaks? -- have a look in the 'calcSlopeDegrees.py'
# do I want to fill seperate arrays?

print "Identifying 'peaks'..."
Unfortunately, after spending a week on this, I've worked out it isn't the method
 I require (will now use from Landserf temporarily) -- however, this may prove useful at some stage

peak_thres = numpy.array([1000., 1500., 2000., 2500., 3000., 3500., 4000.])
peak_drop = 250.
peaks = numpy.zeros(DEM.shape)
peaks2 = numpy.zeros(DEM.shape)
for j in range(1,DEM.shape[1] - 1):
    for i in range(1, DEM.shape[0] - 1):
        neighbours = numpy.array([DEM[i-1,j-1], DEM[i-1,j], DEM[i-1,j+1],
                                  DEM[i,j-1], DEM[i,j+1],
                                  DEM[i+1,j-1], DEM[i+1,j], DEM[i+1,j+1]])
        if (DEM[i,j] >= peak_thres[0,]) and (DEM[i,j] < peak_thres[1,]) and (((DEM[i,j] - neighbours) >= peak_drop).all()):
            peaks[i,j] = 1 #peak_thres[0,]
        elif (DEM[i,j] >= peak_thres[1,]) and (DEM[i,j] < peak_thres[2,]) and (((DEM[i,j] - neighbours) >= peak_drop).all()):
            peaks[i,j] = 1 #peak_thres[1,]
        elif (DEM[i,j] >= peak_thres[2,]) and (DEM[i,j] < peak_thres[3,]) and (((DEM[i,j] - neighbours) >= peak_drop).all()):
            peaks[i,j] = 1 #peak_thres[3,]
        elif (DEM[i,j] >= peak_thres[3,]) and (DEM[i,j] < peak_thres[4,]) and (((DEM[i,j] - neighbours) >= peak_drop).all()):
            peaks[i,j] = 1 #peak_thres[3,]
        elif (DEM[i,j] >= peak_thres[4,]) and (DEM[i,j] < peak_thres[5,]) and (((DEM[i,j] - neighbours) >= peak_drop).all()):
            peaks[i,j] = 1 #peak_thres[4,]
        else:
            peaks[i,j] = 0
print 'Done.'
"""

# Peak density
print 'Calculating peak density per grid cell...'
all_peaks_subset = peaks1000[0:3000, 0:2500]
peak_dens_max, peak_dens_min, peak_dens_range, peak_dens_sum, regions = grid_range(all_peaks_subset, factor)
print 'Done.'

## Calculate elevation range
print 'Calculating elevation range per grid cell...'
DEM_subset = DEM[0:3000, 0:2500]
elev_max, elev_min, elev_range, elev_sum, regions = grid_range(DEM_subset, factor)
# numpy.savetxt(output_folder + 'elev_range.txt', elev_range)
print 'Done.'

# Elevation test for decision tree
#elev_test = elev_range
#elev_test[elev_test>elev_thres_1] = 1 ## may not work -- as two levels?
#
#if elev_test[elev_test>lower]

## Calculate slope range
slope_subset = slope[0:3000,0:2500]
slope_max, slope_min, slope_range, slope_sum, regions = grid_range(slope_subset, factor)
print 'Calculating slope range per grid cell...'
# numpy.savetxt(output_folder + 'slope_range.txt', slope_range)
print 'Done.'

# Slope test for decision tree
#slope_test = slope_range
# need an if loop otherwise will overwrite... 
#slope_test[slope_range>tree_slope_thres] = 1

"""  Do I need to classify these... put into binary, then grid (based upon thresholds above)? """

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
        #fig_hist = plt.figure(i)
        n, bins, patches = plt.hist(hypso_d[~numpy.isnan(hypso_d)], bins=100) # Histo plot, get rid of remaining NaNs
        #plt.show() # can output plots, but slow...!!
        
    ## Skewness test
    skew = stats.skew(hypso_d[~numpy.isnan(hypso_d)]) # perform skewness measure on all but NaN values -- only relevant if not using .all() above
    if skew > 0.1: # Any grid with a skew of greater than 0.1 is classed as positively skewed (change this to threshold up top)
        skewness_test[i,0] = 1
    else:
        skewness_test[i,0] = 0
        
    ## Bimodal test -- are there other ways to do this?
    # Thresholds    
    x_threshold = 0.4 # X-axis (Distance/ Location) threshold -- a percentage of the range of the data, bins further away from this (around peak bin) can be considered separate peaks
    y_threshold = 0.2 # Y-axis (Peak) threshold -- a percentage of the count in the peak bin, bins with more members can be considered peaks
    
    # Tidy data
    bincentres = 0.5*(bins[1:]+bins[:-1]) # find bincenters (as bins produces a 101 length array)
    histo_data = numpy.concatenate(([n],[bincentres])) # joins n (number in bin) with bincentres (bin placement)
    histo_data = numpy.swapaxes(histo_data, 0, 1)
    # Rank histo data -- to find peak (perhaps need to change nomenclature away from 'peak')
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

bimodal_grid=bimodal_test.reshape(p.shape) # need to automate reshape -- perhaps elev_range.shape? (but if thats been done wrong then, could go wrong later)
skewness_grid = skewness_test.reshape(p.shape)

### Decision Tree

"""
is cell in bimodal_grid == 1:
    
    else 
    
    
http://nbviewer.ipython.org/github/gumption/Python_for_Data_Science/blob/master/4_Python_Simple_Decision_Tree.ipynb
"""

# Use ravelled lists...?

# if 

### Plotting
## Inputs
# DEM
fig_DEM = plt.figure(2)
fig_DEM.suptitle('Greenland bedrock GIA DEM', fontsize=12)
plt.imshow(DEM, cmap='terrain', vmax=2000, vmin=-800) # vmax is a way of stretching colour map
plt.xlabel('Distance (km)', fontsize=10)
plt.ylabel('Distance (km)', fontsize=10)
cbar=plt.colorbar(extend='both')
cbar.set_label('Elevation (m)', fontsize=10)
plt.savefig(output_folder + 'DEM_plot.eps', dpi=1200)
print 'DEM plotted successfully -- ' + output_folder + 'DEM_plot.eps'

# Slope
fig_slope = plt.figure(3)
fig_slope.suptitle('Slope of Greenland bedrock (GIA) DEM', fontsize=12)
plt.imshow(slope, cmap='jet_r') # _r is a way to reverse colour map
plt.xlabel('Distance (km)', fontsize=10)
plt.ylabel('Distance (km)', fontsize=10)
cbar=plt.colorbar(extend='neither')
cbar.set_label('slope', fontsize=10)
plt.savefig(output_folder + 'slope_plot.eps', dpi=1200)
print 'Slope plotted successfully -- ' + output_folder + 'slope_plot.eps'

## Outputs
# Identified Peaks
fig_elev_range = plt.figure(4)
fig_elev_range.suptitle('identified_peaks', fontsize=12)
plt.imshow(peaks1000, interpolation='nearest', cmap='Greys', extent=[0,2500,0,3000]) # interpolation 'nearest' stops blurry figures!
plt.xlabel('Distance (km)', fontsize=10)
plt.ylabel('Distance (km)', fontsize=10)
cbar=plt.colorbar(extend='neither')
plt.savefig(output_folder + 'DEM_peaks_plot' + str(factor) + '.eps', dpi=1200)
print 'Identified peaks plotted successfully -- ' + output_folder + 'DEM_peaks_plot.eps'

# Peak Density
fig_elev_range = plt.figure(5)
fig_elev_range.suptitle('peak_density', fontsize=12)
plt.imshow(peak_dens_sum, interpolation='nearest', extent=[0,2500,0,3000]) # interpolation 'nearest' stops blurry figures!
plt.xlabel('Distance (km)', fontsize=10)
plt.ylabel('Distance (km)', fontsize=10)
cbar=plt.colorbar(extend='neither')
cbar.set_label('Elevation range (m)', fontsize=10)
plt.savefig(output_folder + 'DEM_peak_dens' + str(factor) + '.eps', dpi=1200)
print 'Peak density plotted successfully -- ' + output_folder + 'DEM_peak_dens.eps'

"""
need to do mulitple plot with different 'peak' elevations as coloured dots,
then with the peak denisty grid underlain
"""

# Elevation Range
fig_elev_range = plt.figure(6)
fig_elev_range.suptitle('elev_range', fontsize=12)
plt.imshow(elev_range, interpolation='nearest', extent=[0,2500,0,3000]) # interpolation 'nearest' stops blurry figures!
plt.xlabel('Distance (km)', fontsize=10)
plt.ylabel('Distance (km)', fontsize=10)
cbar=plt.colorbar(extend='neither')
cbar.set_label('Elevation range (m)', fontsize=10)
plt.savefig(output_folder + 'DEM_elev_range' + str(factor) + '.eps', dpi=1200)
print 'Elevation range plotted successfully -- ' + output_folder + 'DEM_elev_range.eps'

# Slope Range
fig_slope_range = plt.figure(7)
fig_slope_range.suptitle('slope_range', fontsize=12)
plt.imshow(slope_range, interpolation='nearest', extent=[0,2500,0,3000])
plt.xlabel('Distance (km)', fontsize=10)
plt.ylabel('Distance (km)', fontsize=10)
cbar=plt.colorbar(extend='neither')
cbar.set_label('Slope range (deg)', fontsize=10)
plt.savefig(output_folder + 'DEM_slope_range' + str(factor) + '.eps', dpi=1200)
print 'Slope range plotted successfully -- ' + output_folder + 'DEM_slope_range.eps'

# Binary Skewness (test)
fig_skew = plt.figure(8)
fig_skew.suptitle('skewness', fontsize=12)
plt.imshow(skewness_grid, cmap='Greys', interpolation='nearest', extent=[0,2500,0,3000])
plt.xlabel('Distance (km)', fontsize=10)
plt.ylabel('Distance (km)', fontsize=10)
cbar=plt.colorbar(extend='neither')
plt.savefig(output_folder + 'DEM_skewness_test' + str(factor) + '.eps', dpi=1200)
print 'Skewness mask plotted successfully -- ' + output_folder + 'DEM_skewness_test.eps'

# Binary Modal (test)
fig_bimodal = plt.figure(9)
fig_bimodal.suptitle('bimodal', fontsize=12)
plt.imshow(bimodal_grid, cmap='Greys', interpolation='nearest', extent=[0,2500,0,3000])
plt.xlabel('Distance (km)', fontsize=10)
plt.ylabel('Distance (km)', fontsize=10)
cbar=plt.colorbar(extend='neither')
plt.savefig(output_folder + 'DEM_bimodal_test' + str(factor) + '.eps', dpi=1200)
print 'Bimodal mask plotted successfully -- ' + output_folder + 'DEM_bimodal_test.eps'

# Plot Classified Grid
#fig_classified

# need to set up geolocation/ geotransform

# - create a fishnet of variable size (as in set the size as a variable -- look into fuzzy boxes later)
# - produce a result of elevation range, and slope range (as per fishnet)
# - produce a fishnet with smaller arrays for elevation values per fishnet box
# - produce histograms from fishnet (hypsometry)
# -	work out skewness, and also smooth to work out bimodal??

# plot results!