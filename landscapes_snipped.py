
"""
snip1

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

"""
snip peaks

#input_peaks1500 = input_folder + 'peak_anal/peaks_1500.tif'
#input_peaks2000 = input_folder + 'peak_anal/peaks_2000.tif'
#input_peaks2500 = input_folder + 'peak_anal/peaks_2500.tif'
#input_peaks3000 = input_folder + 'peak_anal/peaks_3000.tif'

#gPeaks1500 = gdal.Open(input_peaks2500, GA_ReadOnly)
#peaks1500 = gPeaks1500.ReadAsArray(0, 0, gPeaks1500.RasterXSize, gPeaks1500.RasterYSize).astype(numpy.float)
#gPeaks2000 = gdal.Open(input_peaks2000, GA_ReadOnly)
#peaks2000 = gPeaks2000.ReadAsArray(0, 0, gPeaks2000.RasterXSize, gPeaks2000.RasterYSize).astype(numpy.float)
#gPeaks2500 = gdal.Open(input_peaks2500, GA_ReadOnly)
#peaks2500 = gPeaks2500.ReadAsArray(0, 0, gPeaks2500.RasterXSize, gPeaks2500.RasterYSize).astype(numpy.float)
#gPeaks3000 = gdal.Open(input_peaks3000, GA_ReadOnly)
#peaks3000 = gPeaks3000.ReadAsArray(0, 0, gPeaks3000.RasterXSize, gPeaks3000.RasterYSize).astype(numpy.float)

#peaks1500[numpy.logical_or(peaks1500 > 5., peaks1500 < 5.)] = 0.
#peaks1500[peaks1500 == 5.] = 1.
#peaks2000[numpy.logical_or(peaks2000 > 5., peaks2000 < 5.)] = 0.
#peaks2000[peaks2000 == 5.] = 1.
#peaks2500[numpy.logical_or(peaks2500 > 5., peaks2500 < 5.)] = 0.
#peaks2500[peaks2500 == 5.] = 1.
#peaks3000[numpy.logical_or(peaks3000 > 5., peaks3000 < 5.)] = 0.
#peaks3000[peaks3000 == 5.] = 1.
"""

"""
snip2

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
