from sklearn import ensemble
from numpy import *
import time
import cPickle
import sys 
from sklearn import preprocessing
from sklearn.externals import joblib

modelname_rf24 = './map_all_model/rf-model-origin24.save'
modelname_svm24 = './map_all_model/svm-model-origin24.save'
modelname_svm48 = './map_all_model/svm-vgg-48-sub128.save'

def load_data(file_name):
  file = open(file_name, 'r')
  line = file.readline()
  datasets = []

  while line:
    datasets.append(line.strip().split(' '))
    line = file.readline()
  datasets = array(datasets)
  return datasets

if __name__ ==  '__main__':
    
  filename_feature = './result_beijing/beijing_vgg128_48f.txt'
  datasets_feature = load_data(filename_feature)
  feature_24 = asfarray(datasets_feature[:, 0:24])
  feature_48 = asfarray(datasets_feature[:, 0:48])
  
  scaler_file_24f = './map_all_model/scaler_24f.save'
  scaler_24f = joblib.load(scaler_file_24f)
  scaler_file_48f = './map_all_model/scaler_48f.save'
  scaler_48f = joblib.load(scaler_file_48f)
    
  feature_24_svm = scaler_24f.transform(feature_24)
  feature_48_svm = scaler_48f.transform(feature_48)

  filename_vggprob = './result_beijing/beijing_vgg128_prob.txt'
  datasets_vggprob = load_data(filename_vggprob)
  datasets_vggprob = asfarray(datasets_vggprob)

  f_rf24 = open(modelname_rf24, 'rb')
  model_rf24 = cPickle.load(f_rf24, encoding='latin1')
  f_rf24.close()

  f_svm24 = open(modelname_svm24, 'rb')
  model_svm24 = cPickle.load(f_svm24, encoding='latin1')
  f_svm24.close()

  f_svm48 = open(modelname_svm48, 'rb')
  model_svm48 = cPickle.load(f_svm48, encoding='latin1')
  f_svm48.close()

  y_prob_rf24 = model_rf24.predict_proba(feature_24)
  y_prob_svm24 = model_svm24.predict_proba(feature_24_svm)
  y_prob_svm48 = model_svm48.predict_proba(feature_48_svm)

  wfile_ensemble = './result_beijing/beijing_svm24_rf24_svm48_vgg_pred.txt'
  fo_ensemble = open(wfile_ensemble, 'a')

  ensemble_probs11 = zeros((11))

  for i in xrange(feature_24.shape[0]):

    ensemble_probs11 = zeros((11))
    rf24_pred = argmax(y_prob_rf24[i])
    svm24_pred = argmax(y_prob_svm24[i])
    list_classes = ['9','10']

    if(rf24_pred in list_classes or svm24_pred in list_classes):
      for k in xrange(11):
        ensemble_probs11[k] = (y_prob_rf24[i][k] + y_prob_svm24[i][k]) * 0.5
      ensemble_pred = argmax(ensemble_probs11)

    else:
      for k in xrange(4):
        ensemble_probs11[k] = (float(y_prob_rf24[i][k]) + float(y_prob_svm24[i][k]) + float(y_prob_svm48[i][k]) + float(datasets_vggprob[i][k])) * 0.25
      ensemble_probs11[7] = (float(y_prob_rf24[i][7]) + float(y_prob_svm24[i][7]) + float(y_prob_svm48[i][4]) + float(datasets_vggprob[i][4])) * 0.25
      ensemble_pred = argmax(ensemble_probs11)

    fo_ensemble.write(str(ensemble_pred) + '\n')
    
  

