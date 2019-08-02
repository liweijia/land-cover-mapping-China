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

specfile = './beijing/LC08_L1TP_123032_20170523_20170526_01_T1_sr_band1-7.tif'
demfile = './beijing/dem_LC81230322016045LGN00_cfmask.tif'
slopefile = './beijing/slope_dem_LC81230322016045LGN0.tif'
maxNImagefile = './beijing/Lpr_123032_image_MaxN.tif'
maxNDoyfile = './beijing/Lpr_123032_doy_MaxN.tif'

mapname = './result_beijing/beijing_rf24.tif'
modelname = './map_all_model/rf-model-origin24.save'

bound = 100000 # the input array size of model.predict in each time, it will run out of memory if the value is too large.
dataset_spec = gdal.Open(specfile)


band_spec = dataset_spec.GetRasterBand(1)
data_spec = band_spec.ReadAsArray(0,0,dataset_spec.RasterXSize,dataset_spec.RasterYSize)
right = data_spec.shape[0]
left =  data_spec.shape[1]

print(right,left)

num = left * right / bound
pi = 3.1415926

def load_data(file_name):
  file = open(file_name, 'r')
  line = file.readline()
  datasets = []

  while line:
    datasets.append(line.strip().split(' '))
    line = file.readline()
  datasets = array(datasets)
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


    def img_pixelXY2xLonyLat(self, xPixArr, yPixArr, imName):
        self.openImage(imName)
        return self.pixelXY2xLonyLat(xPixArr, yPixArr)

    def getWidth(self):

        if self.dtSet is None:
            return None

        return self.dtSet.RasterXSize

    def getHeight(self):

        if self.dtSet is None:
            return None

        return self.dtSet.RasterYSize

### Read pixel value of geotiff image into array
def classify_input():
    #open classify image
    dataset_spec = gdal.Open(specfile)
    dataset_dem = gdal.Open(demfile)
    dataset_slope = gdal.Open(slopefile)
    dataset_maxNImage = gdal.Open(maxNImagefile)
    dataset_maxNDoy = gdal.Open(maxNDoyfile)
    dataset_NDVI = gdal.Open(specfile)
    dataset_maxNDVI = gdal.Open(maxNImagefile)


    #get the spectrum dataset
    classify_input = []
    bandnum_spec = dataset_spec.RasterCount
    for i in range(1,8):
        band_spec = dataset_spec.GetRasterBand(i)
        data_spec = band_spec.ReadAsArray(0,0,dataset_spec.RasterXSize,dataset_spec.RasterYSize)
        data_spec.shape = 1,-1
        classify_input.append(data_spec)
        data_spec = None

    #get xy dataset
    imName = './beijing/LC08_L1TP_123032_20170523_20170526_01_T1_sr_band1-7.tif'

    img = ImageDataIO()
    img.openImage(imName)

    img_width = img.getWidth()
    img_height = img.getHeight()

    xPixArr = zeros(img_width*img_height)
    xPixArr = asfarray(xPixArr)
    yPixArr = zeros(img_width*img_height)
    yPixArr = asfarray(yPixArr)


    k=0
    for i in range(img_height):
      for j in range(img_width):
        xPixArr[k] = j
        yPixArr[k] = i
        k += 1

    xLonYLat = zeros((xPixArr.shape[0], 3), dtype=float64)
    xLonYLat = img.img_pixelXY2xLonyLat(xPixArr,yPixArr,imName)


    begin = img.img_pixelXY2xLonyLat(array(0),array(0),imName)
    end = img.img_pixelXY2xLonyLat(array(right),array(left),imName)
    begin_x = begin[0][0]
    begin_y = begin[0][1]
    end_x = end[0][0]
    end_y = end[0][1]


    #transpose xLonYLat,use the first two lines:
    xLonYLat = xLonYLat.transpose()
    xLonYLat_new1 = zeros((1,xPixArr.shape[0]))
    xLonYLat_new2 = zeros((1,xPixArr.shape[0]))
    for j in range(xPixArr.shape[0]):
      xLonYLat_new1[0][j]=xLonYLat[0][j]
    for j in range(xPixArr.shape[0]):
      xLonYLat_new2[0][j]=xLonYLat[1][j]

    xLonYLat_new1.shape = 1,-1
    xLonYLat_new2.shape = 1,-1
    classify_input.append(xLonYLat_new1)
    classify_input.append(xLonYLat_new2)


    #get dem dataset
    imName = './beijing/dem_LC81230322016045LGN00_cfmask.tif'

    img = ImageDataIO()
    img.openImage(imName)
    begin = img.xLonyLat2PixelXY(begin_x , begin_y , imName)
    end = img.xLonyLat2PixelXY(end_x , end_y , imName)

    band_dem = dataset_dem.GetRasterBand(1)
    data_dem = band_dem.ReadAsArray(begin[0,0],begin[0,1],left,right)
    data_dem.shape = 1,-1
    classify_input.append(data_dem)
    data_dem = None

    #get slope dataset
    imName = './beijing/slope_dem_LC81230322016045LGN0.tif'

    img = ImageDataIO()
    img.openImage(imName)
    begin = img.xLonyLat2PixelXY(begin_x , begin_y , imName)
    end = img.xLonyLat2PixelXY(end_x , end_y, imName)
    band_slope = dataset_slope.GetRasterBand(1)
    data_slope = band_slope.ReadAsArray(begin[0,0],begin[0,1],left,right)

    data_slope.shape = 1,-1
    classify_input.append(data_slope)
    data_slope = None

    #get maxNImage dataset
    imName = './beijing/Lpr_123032_image_MaxN.tif'

    img = ImageDataIO()
    img.openImage(imName)
    begin = img.xLonyLat2PixelXY(begin_x , begin_y , imName)
    end = img.xLonyLat2PixelXY(end_x , end_y , imName)

    bandnum_maxNImage = dataset_maxNImage.RasterCount
    for i in range(1,8):
        band_maxNImage = dataset_maxNImage.GetRasterBand(i)
#        data_maxNImage = band_maxNImage.ReadAsArray(begin[0,0],begin[0,1],end[0,0],end[0,1])
        data_maxNImage = band_maxNImage.ReadAsArray(begin[0,0],begin[0,1],left,right)
        data_maxNImage.shape = 1,-1
        classify_input.append(data_maxNImage)
        data_maxNImage = None

    #get ND dataset
    ND_cos = cos(2 * pi * 308 / 366.) * ones(right * left)
    ND_sin = sin(2 * pi * 308 / 366.) * ones(right * left)
    ND_cos.shape = 1,-1
    ND_sin.shape = 1,-1

    classify_input.append(ND_cos)
    classify_input.append(ND_sin)

    #get maxNDoy dataset
    imName = './beijing/Lpr_123032_doy_MaxN.tif'

    img = ImageDataIO()
    img.openImage(imName)
    begin = img.xLonyLat2PixelXY(begin_x , begin_y , imName)
    end = img.xLonyLat2PixelXY(end_x, end_y , imName)


    band_maxNDoy = dataset_maxNDoy.GetRasterBand(1)
    data_maxNDoy = band_maxNDoy.ReadAsArray(begin[0,0],begin[0,1],left,right)
    data_maxNDoy.shape = 1,-1

    data_maxNDoy_cos = cos(2 * pi * data_maxNDoy / 366.)
    data_maxNDoy_sin = sin(2 * pi * data_maxNDoy / 366.)
    data_maxNDoy_cos.shape = 1,-1
    data_maxNDoy_sin.shape = 1,-1
    classify_input.append(data_maxNDoy_cos)
    classify_input.append(data_maxNDoy_sin)
    data_maxNDoy = None

    #get NDVI dataset
    bandnum_NDVI = dataset_NDVI.RasterCount
    band_NDVI_4 = dataset_NDVI.GetRasterBand(4)
    data_NDVI_4 = band_NDVI_4.ReadAsArray(0,0,dataset_NDVI.RasterXSize,dataset_NDVI.RasterYSize)
    data_NDVI_4.shape = 1,-1
    band_NDVI_5 = dataset_NDVI.GetRasterBand(5)
    data_NDVI_5 = band_NDVI_5.ReadAsArray(0,0,dataset_NDVI.RasterXSize,dataset_NDVI.RasterYSize)
    data_NDVI_5.shape = 1,-1

    data_NDVI = zeros(data_NDVI_4.shape[1])
    for i in range(data_NDVI_4.shape[1]):
      if (data_NDVI_4[0][i] + data_NDVI_5[0][i] == 0):
        data_NDVI[i] = 0
      else:
        data_NDVI[i] = (data_NDVI_5[0][i] - data_NDVI_4[0][i]) * 1.0 / (data_NDVI_4[0][i] + data_NDVI_5[0][i])
    data_NDVI.shape = 1,-1
    classify_input.append(data_NDVI)

    #get maxNDVI dataset
    imName = './beijing/Lpr_123032_image_MaxN.tif'

    img = ImageDataIO()
    img.openImage(imName)
    begin = img.xLonyLat2PixelXY(begin_x , begin_y , imName)
    end = img.xLonyLat2PixelXY(end_x , end_y , imName)

    bandnum_maxNDVI = dataset_maxNDVI.RasterCount
    band_maxNDVI_4 = dataset_maxNDVI.GetRasterBand(4)
    data_maxNDVI_4 = band_maxNDVI_4.ReadAsArray(begin[0,0],begin[0,1],left,right)
    data_maxNDVI_4.shape = 1,-1
    band_maxNDVI_5 = dataset_maxNDVI.GetRasterBand(5)
    data_maxNDVI_5 = band_maxNDVI_5.ReadAsArray(begin[0,0],begin[0,1],left,right)
    data_maxNDVI_5.shape = 1,-1
    data_maxNDVI = zeros(data_maxNDVI_4.shape[1])
    for i in range(data_maxNDVI_4.shape[1]):
      if (data_maxNDVI_4[0][i] + data_maxNDVI_5[0][i] == 0):
        data_maxNDVI[i] = 0
      else:
        data_maxNDVI[i] = (data_maxNDVI_5[0][i] - data_maxNDVI_4[0][i]) * 1.0 / (data_maxNDVI_4[0][i] + data_maxNDVI_5[0][i])
    data_maxNDVI.shape = 1,-1
    classify_input.append(data_maxNDVI)

    #get classification features(24)
    classify_input = array(classify_input)
    classify_input.shape = classify_input.shape[0],-1
    classify_input = transpose(classify_input)
    classify_input = array(classify_input)


    wfile1 = 'beijing_before_24f.txt'
    fo1 = open(wfile1, 'a')
    for i in range(classify_input.shape[0]):
      for j in range(classify_input.shape[1]):
        fo1.write(str(classify_input[i][j]) + ' ')
      fo1.write('\n')

    fo1.close()

    return classify_input

### Convert array of predited labels into geotiff image
def classify_output(label_pred):

    #label_pred = transpose(label_pred)
    label_pred = array(label_pred)
    #write mapdata
    dataset_spec = gdal.Open(specfile)
    driver = dataset_spec.GetDriver()
    mapDataset = driver.Create(mapname, dataset_spec.RasterXSize, dataset_spec.RasterYSize, 1, GDT_Int16)
    mapBand = mapDataset.GetRasterBand(1)
    mapBand.WriteArray(label_pred, 0, 0)

    #write spatial references
    geoTransform = dataset_spec.GetGeoTransform()
    mapDataset.SetGeoTransform(geoTransform)
    projection = dataset_spec.GetProjection()
    mapDataset.SetProjection(projection)

    label_assumed = None
    mapDataset = None
    gc.collect()

def mapping():
    ### Read RF model
    f = open(modelname, 'rb')
    model = cPickle.load(f, encoding='latin1')
    f.close()


    ### Read geotiff image
    start_time = time.clock()
    glc_input = classify_input()
    end_time = time.clock()
    print ("Load tiff time = ", (end_time - start_time) / 60.0)

    ### classify_output
    start_time = time.clock()
    tmp_pred = ndarray([bound, num])
    tmp_input = ndarray([bound, 24])
    res_pred = ndarray([glc_input.shape[0] - num*bound, 1])
    res_input = ndarray([glc_input.shape[0] - num*bound, 24])

    for i in range(num):
      for j in range(0, bound):
        tmp_input[j] = glc_input[i*bound + j]
      tmp_pred[:, i] = model.predict(tmp_input)

    for j in range(0, glc_input.shape[0] - num*bound):
      res_input[j] = glc_input[num*bound + j]
    res_pred[:, 0] = model.predict(res_input)


    # convert tmp_pred and res_pred into label_pred
    label_pred = ndarray([right, left])
    for i in range(right*left):
      if i/bound < num:
        label_pred[i/left, i%left] = tmp_pred[i%bound, i/bound]
      else:
        label_pred[i/left, i%left] = res_pred[i-num*bound, 0]

    label_pred = array(label_pred)


    wfile2 = 'beijing_before_rf_label.txt'
    fo2 = open(wfile2, 'a')
    for i in range(label_pred.shape[0]):
      for j in range(label_pred.shape[1]):
        fo2.write(str(label_pred[i][j]) + '\n')
    end_time = time.clock()
    print ("predict time = ", (end_time - start_time) / 60.0)

    start_time = time.clock()
    classify_output(label_pred)
    end_time = time.clock()
    print "mapping time = ", (end_time - start_time) / 60.0

if __name__ == '__main__':
    mapping()


