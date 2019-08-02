from sklearn import ensemble
from numpy import *
import time
import pandas as pd
import cPickle
import sys
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.svm import SVC

### the top_left coordinate and bottom_right coordinate of GE-HRI relative to Landsat image
map_start_x = 1942
map_start_y = 3981
map_end_x = 3937
map_end_y = 6000

landsat_width = 7751
ge_width = map_end_x - map_start_x
ge_height = map_end_y - map_start_y

def load_data(file_name):
  file = open(file_name, 'rb')
  line = file.readline()
  datasets = []

  while line:
    datasets.append(line.strip().split(' '))
    line = file.readline()
  datasets = array(datasets)
  return datasets

if __name__ ==  '__main__':
  filename_24f = 'beijing_before_24f.txt'
  datasets_24f = load_data(filename_24f)

  filename_cnnfc7 = 'beijing_vgg128_fc_24f.txt'
  datasets_cnnfc7 = pd.read_table(filename_cnnfc7,header=None,sep=' ')
  datasets_cnnfc7 = array(datasets_cnnfc7)
  all_nan = isnan(datasets_cnnfc7)
  datasets_cnnfc7[all_nan] = 9999

  wfile_cnnfc7 = 'beijing_vgg128_48f.txt'
  fo_cnnfc7 =  open(wfile_cnnfc7, 'a')

  begin = 0
  for i in range(ge_height):
    for j in range(begin,begin+ge_width):
      for k in range(0, datasets_24f.shape[1]):
        fo_cnnfc7.write(str(datasets_24f[j][k]) + ' ')
      for k in range(0, datasets_cnnfc7.shape[1]):
        fo_cnnfc7.write(str(datasets_cnnfc7[i*ge_width+(j-begin)][k]) + ' ')
      fo_cnnfc7.write('\n')
    begin = begin + landsat_width
