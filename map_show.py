from sklearn import ensemble
import time
import gdal
from numpy import *
from gdalconst import *
import random
import gc
import cPickle
import sys
import osr
from PIL import Image
import cv2

import numpy as np
import caffe
import os

mapname = 'map_beijing_svm24_rf24_svm48_vgg_pred.tif'
imName_spec = './beijing/LC08_L1TP_123032_20170523_20170526_01_T1_sr_band1-7.tif'
readname = './result_beijing/beijing_svm24_rf24_svm48_vgg_pred.txt'

#beijing
map_start_x = 1942
map_start_y = 3981
map_end_x = 3937
map_end_y = 6000

number = (map_end_x - map_start_x) * (map_end_y - map_start_y)

def load_data(file_name):
  file = open(file_name, 'rb')
  line = file.readline()
  datasets = []

  while line:
    datasets.append(line.strip().split(' '))
    line = file.readline()
  datasets = np.array(datasets)
  return datasets

def coord_PTs_ProjGeo2Pix(xLonyLat, geoTrans, xyPix):
    """
           projected longitude latitude -->  xyPix;   equal length
    """

    xyPix[:,0] = array(floor((xLonyLat[:,0] - geoTrans[0]) / geoTrans[1]), dtype=int64)
    xyPix[:,1] = array(floor((xLonyLat[:,1] - geoTrans[3]) / geoTrans[5]), dtype=int64)

def coord_PTs_Pix2ProjGeo(xyPix, geoTrans, xLonyLat):
    """
           xyPix --> projected longitude latitude;   equal length
    """
    xLonyLat[:, 0] = (xyPix[:, 0] + 0.5) * geoTrans[1] + geoTrans[0]
    xLonyLat[:, 1] = (xyPix[:, 1] + 0.5) * geoTrans[5] + geoTrans[3]


class ImageDataIO(object):
    """
    classdocs
    """
    '''
    gdal.AllRegister will be executed only one time since the process starting
    '''
    gdal.AllRegister()

    def __init__(self):
        """
        Constructor
        """
        self.dtSet = None
        self.outDtSet = None

    def closeImage(self):
        self.dtSet = None

    def openImage(self, imName):

        return self.initReadingImage(imName)
    def initReadingImage(self, imName):

        self.closeImage()
        try:
            self.dtSet = gdal.Open(imName)
        # srcProj = self.dtSet.GetProjection()
        #             print(srcProj)
        except Exception as e:
            logging.error('open: ' + imName)
            logging.error('Reason: ' + str(e))
            return False

        if self.dtSet is None:
            logging.error('null data set when open: ' + imName)
            logging.error(gdal.GetLastErrorMsg())
            return False

        return True

    def pixelXY2xLonyLat(self, xPixArr, yPixArr):

        if self.dtSet is None:
            logging.error("image were not opened when converting coordinates")
            return None

        geoTrans = self.dtSet.GetGeoTransform()

        xyPix = hstack((xPixArr.reshape(-1, 1), yPixArr.reshape(-1, 1)))
        projXLonYLat = xyPix.astype(float64)
        coord_PTs_Pix2ProjGeo(xyPix, geoTrans, projXLonYLat)

        strProj = self.dtSet.GetProjection()

        srim = osr.SpatialReference()
        srim.ImportFromWkt(strProj)

        srgeo = srim.CloneGeogCS()
        coordtransform = osr.CreateCoordinateTransformation(srim, srgeo)

        xLonYLat = zeros((projXLonYLat.shape[0], 3), dtype=float64)
        xLonYLat[:, :] = coordtransform.TransformPoints(projXLonYLat)

        return xLonYLat  # numpy.array(xLonYLat)

    def xLonyLat2PixelXY(self, xLonArr, yLatArr, imName):

        if self.dtSet is None:
            logging.error("image were not opened when converting coordinates")
            return None

        geoTrans = self.dtSet.GetGeoTransform()
        strProj = self.dtSet.GetProjection()

        srim = osr.SpatialReference()
        srim.ImportFromWkt(strProj)

        srgeo = srim.CloneGeogCS()
        coordtransform = osr.CreateCoordinateTransformation(srgeo, srim)

        xLonYLat = vstack((xLonArr, yLatArr))
        #print xLonYLat

        projXLonYLat = coordtransform.TransformPoints(xLonYLat.T)

        xyPix = xLonYLat.T.astype(int64)
        coord_PTs_ProjGeo2Pix(array(projXLonYLat), geoTrans, xyPix)
        #print numpy.array(projXLonYLat)
        return xyPix


    def getWidth(self):

        if self.dtSet is None:
            return None

        return self.dtSet.RasterXSize

    def getHeight(self):

        if self.dtSet is None:
            return None

        return self.dtSet.RasterYSize



### Convert array of predited labels into geotiff image
def classify_output(label_pred):

    #label_pred = transpose(label_pred)
    map_width = map_end_x - map_start_x
    map_height = map_end_y - map_start_y
    specfile = './beijing/LC08_L1TP_123032_20170523_20170526_01_T1_sr_band1-7.tif'
    #write mapdata
    dataset_spec = gdal.Open(specfile)
    driver = dataset_spec.GetDriver()
    spec_width = dataset_spec.RasterXSize
    spec_height = dataset_spec.RasterYSize

    mapDataset = driver.Create(mapname, spec_width, spec_height, 1, GDT_Int16)
    mapBand = mapDataset.GetRasterBand(1)

    ###convert label_pred array to label_output array
    label_output = ndarray([spec_height, spec_width])


    ### array initalization
    for i in range(spec_height):
        for j in range(spec_width):
            label_output[i,j] = 99

    ### assign values for label_output if (i,j) is in map
    for i in xrange(map_start_y, map_end_y):
        for j in xrange(map_start_x, map_end_x):
            i_in_map = i - map_start_y
            j_in_map = j - map_start_x
            index_in_map = i_in_map * map_width + j_in_map
            label_output[i,j] = label_pred[index_in_map]

    #label_pred.shape = map_height, map_width
    #label_output = array(label_pred)
    mapBand.WriteArray(label_output,0,0)

    #write spatial references
    geoTransform = dataset_spec.GetGeoTransform()
    mapDataset.SetGeoTransform(geoTransform)
    projection = dataset_spec.GetProjection()
    mapDataset.SetProjection(projection)


def mapping():
    array_pred = load_data(readname)
    classify_output(array_pred)

if __name__ == '__main__':
    mapping()


