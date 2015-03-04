# script by Michael A. Cooper (@macooperr) on 4th Mar 2015 : 18:12 GMT
#
# produce classification of 'landscapes of glacial erosion' from DEMs
#

import gdal
from gdalconst import *

import numpy

DEM = '/Users/mc14909/Dropbox/Bristol/data/nuuk_mac/GIA_lip/GIA_bedDEM.tif'
dataset = gdal.Open(DEM, GA_ReadOnly)

cols = dataset.RasterXSize
rows = dataset.RasterYSize
driver = dataset.GetDriver().LongName

# need to set up geolocation/ geotransform

data = dataset.ReadasArray(0, 0, cols, rows).astype(numpy.float)

# work out how to plot this array as surface plot
# work out how to:
# - create a 'slope' image
# - create a fishnet of variable size (as in set the size as a variable -- look into fuzzy boxes later)
# - produce a result of elevation range, and slope range (as per fishnet)
# - produce a fishnet with smaller arrays for elevation values per fishnet box
# - produce histograms from fishnet (hypsometry)
# 		-	work out skewness, and also smooth to work out bimodal??

# plot results!

