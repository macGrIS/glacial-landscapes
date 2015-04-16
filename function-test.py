# grid function test
from __future__ import division

import gdal
from gdalconst import *
import numpy
from scipy import ndimage
from scipy import stats
from matplotlib import pyplot as plt
import os, sys, shutil


input_folder = '/Users/mc14909/Dropbox/Bristol/data/glacial-landscapes/'
inputDEM = input_folder + 'GIA_bedDEM_clipped2.tif'

factor = 100

gDEM = gdal.Open(inputDEM, GA_ReadOnly)
DEM = gDEM.ReadAsArray(0, 0, gDEM.RasterXSize, gDEM.RasterYSize).astype(numpy.float)
DEMnull = DEM[0,0] # input no data value for tiff
a = numpy.where(DEM == DEMnull) # finds all values in array DEM equal to nodata_value
DEM[a] = numpy.NaN # set nodata_value to NaN

DEM_subset = DEM[0:3000, 0:2500]

array = DEM_subset

assert isinstance(factor, int), type(factor)
# need an assert statement to catch whether the factor is divisible by shape
sy, sx = array.shape
Y, X = numpy.ogrid[0:sy, 0:sx] # defines X Y axes -- not an entire grid
regions = sx/factor * (Y/factor) + (X/factor) # ensure that sx is longest axis of array -- make it run the largest array
block_max = ndimage.maximum(array, labels=regions, index=numpy.arange(regions.max() + 1)) # need to remove the +1 ??
block_max.shape = (sy/factor, sx/factor)
block_min = ndimage.minimum(array, labels=regions, index=numpy.arange(regions.max() + 1))
block_min.shape = (sy/factor, sx/factor)
block_range = block_max - block_min
block_sum = ndimage.sum(array, labels=regions, index=numpy.arange(regions.max() + 1))
block_sum.shape = (sy/factor, sx/factor)
