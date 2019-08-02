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

imName_spec = './beijing/LC08_L1TP_123032_20170523_20170526_01_T1_sr_band1-7.tif'
imName_ge = './beijing/beijing-2000-4000-4000-6000-20170524.tif'

### the top_left coordinate and bottom_right coordinate of GE-HRI relative to Landsat image
map_start_x = 1942
map_start_y = 3981
map_end_x = 3937
map_end_y = 6000
number = (map_end_x - map_start_x) * (map_end_y - map_start_y)

def load_data(file_name):
  file = open(file_name, 'r')
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

def VGG_predict(datasets_val):
    #caffe_root = '/home/lwj/.conda/envs/caffe/user-caffe/'
    caffe_root = './'
    
    sys.path.insert(0, caffe_root + '../bin')

    caffe.set_mode_gpu()
    #caffe.set_mode_cpu()

    #model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
    model_def = caffe_root + './vgg_train_file/VGG_ILSVRC_16_layers_deploy.prototxt'
    #model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    model_weights = caffe_root + './vgg_train_file/caffe_vgg_sub128_train_5_iter_4240.caffemodel'

    net = caffe.Net(model_def,    # defines the structure of the model
                  model_weights,  # contains the trained weights
                  caffe.TEST)     # use test mode (e.g., don't perform dropout)
    MEAN_PROTO_PATH = caffe_root + './vgg_train_file/china_train5_sub128_mean.binaryproto'
    MEAN_NPY_PATH = caffe_root + './vgg_train_file/china_train5_sub128_mean.npy'
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(MEAN_PROTO_PATH, 'rb' ).read()
    blob.ParseFromString(data)
    array = np.array(caffe.io.blobproto_to_array(blob))
    mean_npy = array[0]
    np.save(MEAN_NPY_PATH ,mean_npy)

    #mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    mu = np.load(caffe_root + './vgg_train_file/china_train5_sub128_mean.npy')
    #mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    #transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    #transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

    wfile = 'beijing_vgg128_fc_24f.txt'  ### save 24 features in fc7_tune layer (will be used in SVM-fusion classifier)
    fo = open(wfile, 'a')

    wfile_pred = 'beijing_vgg128_pred_label.txt'  ### save labels predicted from vgg classifier
    fo_pred = open(wfile_pred, 'a')

    wfile_prob = 'beijing_vgg128_prob.txt' ### save probabilities of each type predicted from vgg classifier
    fo_prob = open(wfile_prob, 'a')

    array_pred = ndarray([datasets_val.shape[0]])
#    print datasets_val.shape[0],number

    ### datasets_val [num_pixel, img_size, img_size, 3]
    for i in xrange(datasets_val.shape[0]):
      ### load the image
      image = caffe.io.load_array(datasets_val[i])
      #image = caffe.io.load_image(datasets_val[i])
      #label = datasets_val[i][1]

      # copy the image data into the memory allocated for the net
      transformed_image = transformer.preprocess('data', image)
      net.blobs['data'].data[...] = transformed_image

      # perform classification
      output = net.forward()

      output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

      ### get predicted labels
      pred_label = output_prob.argmax()
      #fo_pred.write(str(label) + ' ' + str(pred_label) + '\n')
      fo_pred.write(str(pred_label) + '\n')
      for prob in range(5):
        fo_prob.write(str(output_prob[prob]) + ' ')
      fo_prob.write('\n')
      array_pred[i] = pred_label

      ### get midoutput
      feat = net.blobs['fc7_tune'].data[0]

      #fo.write(label + ' ')
      for j in xrange(feat.shape[0]):
        fo.write(str(feat[j]) + ' ')
      fo.write('\n')

    fo_pred.close()
    fo.close()

    return array_pred

def classify_input():
    #open classify image
    img_spec = ImageDataIO()
    img_spec.openImage(imName_spec)
    img_width = img_spec.getWidth()
    img_height = img_spec.getHeight()

    img_ge = ImageDataIO()
    img_ge.openImage(imName_ge)
    img_ge_width  = img_ge.getWidth()
    img_ge_height = img_ge.getHeight()

    #get the spectrum dataset
    classify_input = []
    bandnum_spec = img_spec.dtSet.RasterCount
    for i in range(1,2):
        band_spec = img_spec.dtSet.GetRasterBand(i)
        data_spec = band_spec.ReadAsArray(0,0,img_width,img_height)
        data_spec.shape = 1,-1
        classify_input.append(data_spec)
        data_spec = None


    ### the top_left coordinate and bottom_right coordinate of GE-HRI relative to Landsat image
    x_start = 1942
    y_start = 3981
    x_end = 3937
    y_end = 6000


    spec_ge_width  = x_end - x_start
    spec_ge_height = y_end - y_start
    array_length = spec_ge_width * spec_ge_height
    xPixArr = zeros(array_length)
    xPixArr = asfarray(xPixArr)
    yPixArr = zeros(array_length)
    yPixArr = asfarray(yPixArr)

    k=0
    for i in range(y_start, y_end):
      for j in range(x_start, x_end):
        xPixArr[k] = j
        yPixArr[k] = i
        k += 1

    xLonYLat = zeros((xPixArr.shape[0], 3), dtype=float64)
    xLonYLat = img_spec.pixelXY2xLonyLat(xPixArr,yPixArr)


    wfile_xy = 'beijing_vgg128_xy.txt'
    fo_xy = open(wfile_xy, 'a')
    for i in range(xLonYLat.shape[0]):
      fo_xy.write(str(xLonYLat[i][0]) + ' '  + str(xLonYLat[i][1])  + '\n')

    band_ge_r = img_ge.dtSet.GetRasterBand(1)
    band_ge_g = img_ge.dtSet.GetRasterBand(2)
    band_ge_b = img_ge.dtSet.GetRasterBand(3)

    xyPix_ge = img_ge.xLonyLat2PixelXY(xLonYLat[:,0], xLonYLat[:,1], imName_ge)
    img_size = 128
    n_pixels = 2019 ### map_end_y - map_start_y
    n_start = 0
    image_array_ge = zeros((n_pixels, img_size, img_size, 3))
    image_path_array = []
    array_pred_all = []

    #for i in xrange(xyPix_ge.shape[0]):
    for step in xrange((number-n_start)/n_pixels):
      for i in xrange(n_start,n_start+n_pixels):
        center_x = xyPix_ge[i,0]
        center_y = xyPix_ge[i,1]
        j_start = center_y - img_size/2 # global y_cord in ge image
        k_start = center_x - img_size/2 # global x_cord in ge image
        #xLonYLat_ge = img_ge.pixelXY2xLonyLat(center_x, center_y)
        #print xLonYLat[i],xLonYLat_ge

        image_array_ge[i-n_start,:,:,2] = band_ge_r.ReadAsArray(k_start, j_start, img_size, img_size)
        image_array_ge[i-n_start,:,:,1] = band_ge_g.ReadAsArray(k_start, j_start, img_size, img_size)
        image_array_ge[i-n_start,:,:,0] = band_ge_b.ReadAsArray(k_start, j_start, img_size, img_size)

      ### predict label for each element in image_array_ge
      array_pred = VGG_predict(image_array_ge)
      array_pred_all.append(array_pred)
      n_start += n_pixels
    return array_pred_all


def mapping():
    array_pred = classify_input()

if __name__ == '__main__':
    mapping()


