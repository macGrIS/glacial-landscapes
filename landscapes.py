# -*- coding: utf-8 -*-\
from __future__ import division
"""
	4th March, 2015; 18:12 GMT
	Script by Michael A. Cooper (t: @macooperr; git: @macGrIS)
	to produce classification maps of 'landscapes of glacial erosion' (as per Ja
	mieson, et al., 2014; Sugden and John, 1975) from GeoTIFF DEMs.
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
# scipy.ndimage used to calculate statistics (min, max, range and sum) of values (array) for different size grids (factor)
def grid_range(array, factor):
    assert isinstance(factor, int), type(factor)
    # need an assert statement to catch whether the factor is divisible by shape
    sy, sx = array.shape
    Y, X = numpy.ogrid[0:sy, 0:sx] # defines X Y axes -- not an entire grid
    regions = sx/factor * (Y+1/factor) + (X+1/factor) # ensure that sx is longest axis of array -- make it run the largest array
    block_max = ndimage.maximum(array, labels=regions, index=numpy.arange(regions.max() + 1)) # need to remove the +1 ??
    block_max.shape = (sy/factor, sx/factor)
    block_min = ndimage.minimum(array, labels=regions, index=numpy.arange(regions.max() + 1))
    block_min.shape = (sy/factor, sx/factor)
    block_range = block_max - block_min
    block_sum = ndimage.sum(array, labels=regions, index=numpy.arange(regions.max() + 1))
    block_sum.shape = (sy/factor, sx/factor)
    return block_max, block_min, block_range, block_sum, regions;

### Ask user for deets...??
### Define input
input_folder = '/Users/mc14909/Dropbox/Bristol/data/glacial-landscapes/'
output_folder = '/Users/mc14909/Dropbox/Bristol/scratch/glacial-landscapes/'
inputDEM = input_folder + 'GIA_bedDEM_clipped2.tif'
inputSlope = input_folder + 'GDAL_slope2.tif' # must be a way to calculate this (see below), until then, using gdaldem slope output

### Define parameters and metrics
## size of grid/ fishnet (in km -- for 1km posting data) -- LOOK INTO FUZZY BOXES LATER
factor = 100 # doesn't seem to work for 150, or 200km grid sizes (why??) # not actually a factor -- call this target grid size
print 'Grid cell size = ' + str(factor) + ' km.' # units depend on pixel 'size'

## Input subsets (specific catchments?) - "do you wish to subset the data?" (perhaps at end...?)
#local_factor = 50 # factor of subset

# Decision Tree (based upon Jamieson, et al. 2014) -- Change for 'sensitivity analysis' -- ask for custom, or standard.
# Elevation range threshold lower
tree_elev_lower_thres = 1000. # orig < 1000 m
# Elevation range threshold upper
tree_elev_upper_thres = 2000. # orig >= 1000 and < 2000 m
# Peak density threshold
tree_pd_thres = 20. # orig <= 20
# Slope threshold
tree_slope_thres = 5. # orig <= 5

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
p = numpy.zeros([peaky/factor,peakx/factor]) # new grid of factor size size -- to fill with metric data

### Calculate metrics
## Calculate slope -- need to add in this (possibly solved through convolution)
print 'Calculating slope...'
# snip1

print 'Done.'
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

## Peak analysis
# Open peak detection results from Landserf -- value of 5 depicts summit (to use for peak density)
# Currently only using peaks1000 (as it displays ALL peaks over 1000 m in elevation) -- may require others later (snip peaks)
input_peaks1000 = input_folder + 'peak_anal/peaks_1000.tif'

gPeaks1000 = gdal.Open(input_peaks1000, GA_ReadOnly)
peaks1000 = gPeaks1000.ReadAsArray(0, 0, gPeaks1000.RasterXSize, gPeaks1000.RasterYSize).astype(numpy.float)

peaks1000[numpy.logical_or(peaks1000 > 5., peaks1000 < 5.)] = 0.
peaks1000[peaks1000 == 5.] = 1.

## Identify peaks - broken methodology -- maybe try local maximums?
# snip2

## Calculate peak density
print 'Calculating peak density per grid cell...'
all_peaks_subset = peaks1000[0:3000, 0:2500]
peak_dens_max, peak_dens_min, peak_dens_range, peak_density, regions = grid_range(all_peaks_subset, factor)
print 'Done.'

## Calculate elevation range
print 'Calculating elevation range per grid cell...'
DEM_subset = DEM[0:3000, 0:2500]
elev_max, elev_min, elev_range, elev_sum, regions = grid_range(DEM_subset, factor)
# numpy.savetxt(output_folder + 'elev_range.txt', elev_range)
print 'Done.'

## Calculate slope range
slope_subset = slope[0:3000,0:2500]
slope_max, slope_min, slope_range, slope_sum, regions = grid_range(slope_subset, factor)
print 'Calculating slope range per grid cell...'
# numpy.savetxt(output_folder + 'slope_range.txt', slope_range)
print 'Done.'

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
        #print 'Too many NaN values for cell: ' + str(i) + ', skipping...'
        continue
    else:
        #fig_hist = plt.figure(1) # have this as plt.figure(i)? then below as i+1 etc.
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

### Decision Tree -- have to nest logical_and commands as they do not accept more than 2 arguments.
landscape_gd = numpy.zeros(p.shape)
# Areal Scour
landscape_gd[elev_range < tree_elev_lower_thres] = 1.
landscape_gd[numpy.logical_and(numpy.logical_and(elev_range >= tree_elev_lower_thres, elev_range < tree_elev_upper_thres), numpy.logical_and(bimodal_grid == 0., slope_range > tree_slope_thres))] = 1.
# Selective Linear Erosion
landscape_gd[numpy.logical_and(numpy.logical_and(elev_range >= tree_elev_lower_thres, elev_range < tree_elev_upper_thres), bimodal_grid == 1.)] = 2.
landscape_gd[numpy.logical_and(elev_range >= tree_elev_upper_thres, skewness_grid == 1.)] = 2.
landscape_gd[numpy.logical_and(numpy.logical_and(elev_range >= tree_elev_upper_thres, skewness_grid == 0.), peak_density <= tree_pd_thres)] = 2.
# Mainly Alpine
landscape_gd[numpy.logical_and(numpy.logical_and(elev_range >= tree_elev_upper_thres, skewness_grid == 0.), peak_density > tree_pd_thres)] = 3.

### Plotting -- make function for plots, save lines of code
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
fig_id_peaks = plt.figure(4)
fig_id_peaks.suptitle('identified_peaks', fontsize=12)
plt.imshow(peaks1000, interpolation='nearest', cmap='Greys', extent=[0,2500,0,3000]) # interpolation 'nearest' stops blurry figures!
plt.xlabel('Distance (km)', fontsize=10)
plt.ylabel('Distance (km)', fontsize=10)
cbar=plt.colorbar(extend='neither')
plt.savefig(output_folder + 'DEM_peaks_plot' + str(factor) + '.eps', dpi=1200)
print 'Identified peaks plotted successfully -- ' + output_folder + 'DEM_peaks_plot.eps'

# Peak Density
fig_peak_density = plt.figure(5)
fig_peak_density.suptitle('peak_density', fontsize=12)
plt.imshow(peak_density, interpolation='nearest', extent=[0,2500,0,3000]) # extent changes the extent of image (so axes read correctly) -- should probably automate this (for varying size arrays)
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
plt.imshow(elev_range, interpolation='nearest', extent=[0,2500,0,3000])
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

## Result
# Plot Classified Grid
fig_classified = plt.figure(10)
fig_classified.suptitle('landscapes', fontsize=12)
plt.imshow(landscape_gd, interpolation='nearest', extent=[0,2500,0,3000])
plt.xlabel('Distance (km)', fontsize=10)
plt.ylabel('Distance (km)', fontsize=10)
cbar=plt.colorbar(extend='neither')
plt.savefig(output_folder + 'DEM_landscapes_' + str(factor) + 'km.eps', dpi=1200)
print 'Classified landscapes plotted successfully -- ' + output_folder + 'DEM_landscapes.eps'
