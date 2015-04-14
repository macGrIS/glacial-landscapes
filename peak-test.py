# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:31:05 2015

@author: mc14909
"""

import gdal
from gdalconst import *
import numpy
from scipy import ndimage
from scipy import stats
from matplotlib import pyplot as plt
import os, sys, shutil

test1 = numpy.array([[1., 1., 1.],[1.,1000.,1.],[1.,1.,1.]])
test2 = numpy.array([[1000., 1000., 1000.],[1000.,1000.,1000.],[1000.,1000.,1000.]])
test3 = numpy.array([[750., 750., 750.],[750.,1000.,750.],[750.,750.,750.]])
test4 = numpy.array([[1000., 1000., 1000.],[1000.,1.,1000.],[1000.,1000.,1000.]])

peak_thres = numpy.array([1000., 1500., 2000., 2500., 3000., 3500., 4000.])
peak_drop = 250.
peaks1 = numpy.zeros(test1.shape)
peaks2 = numpy.zeros(test1.shape)
peaks3 = numpy.zeros(test1.shape)
peaks4 = numpy.zeros(test1.shape)

i = 1
j = 1

neighbours = numpy.array([test1[i-1,j-1], test1[i-1,j], test1[i-1,j+1], test1[i,j-1], test1[i,j+1], test1[i+1,j-1], test1[i+1,j], test1[i+1,j+1]])

# test1 - peak
if (test1[i,j] >= peak_thres[0,]) and (test1[i,j] < peak_thres[1,]) and (((test1[i,j] - neighbours) >= peak_drop).all()):
    peaks1[i,j] = 1 #peak_thres[0,]
else:
    peaks1[i,j] = 0

neighbours = numpy.array([test2[i-1,j-1], test2[i-1,j], test2[i-1,j+1], test2[i,j-1], test2[i,j+1], test2[i+1,j-1], test2[i+1,j], test2[i+1,j+1]])

# test2 - peak
if (test2[i,j] >= peak_thres[0,]) and (test2[i,j] < peak_thres[1,]) and (((test2[i,j] - neighbours) >= peak_drop).all()):
    peaks2[i,j] = 1 #peak_thres[0,]
else:
    peaks2[i,j] = 0


neighbours = numpy.array([test3[i-1,j-1], test3[i-1,j], test3[i-1,j+1], test3[i,j-1], test3[i,j+1], test3[i+1,j-1], test3[i+1,j], test3[i+1,j+1]])

# test3 - peak
if (test3[i,j] >= peak_thres[0,]) and (test3[i,j] < peak_thres[1,]) and (((test3[i,j] - neighbours) >= peak_drop).all()):
    peaks3[i,j] = 1 #peak_thres[0,]
else:
    peaks3[i,j] = 0



neighbours = numpy.array([test4[i-1,j-1], test4[i-1,j], test4[i-1,j+1], test4[i,j-1], test4[i,j+1], test4[i+1,j-1], test4[i+1,j], test4[i+1,j+1]])

# test4 - peak
if (test4[i,j] >= peak_thres[0,]) and (test4[i,j] < peak_thres[1,]) and (((test4[i,j] - neighbours) >= peak_drop).all()):
    peaks4[i,j] = 1 #peak_thres[0,]
else:
    peaks4[i,j] = 0