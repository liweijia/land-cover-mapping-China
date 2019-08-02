
Integrating Google Earth imagery with Landsat data to improve 30-m land cover mapping in China

Original Feature extraction
glc_RF_mapping_24f.py
Input：
- Landsat images
- rf-model-origin24.save
Output：
- beijing_before_24f.txt

VGG based land cover mapping and feature extraction
glc_vgg_mapping_whole.py
Input：
- VGG model
- Google Earth high-resolution images，Landsat images
Output：
- beijing_vgg128_fc_24f.txt
- beijing_vgg128_prob.txt

Combination of 24 original features and 24 features extracted by VGG
combine_48f.py
Input:
- beijing_before_24f.txt
- beijing_vgg128_fc_24f.txt
Output：
- beijing_vgg128_48f.txt

RF/SVM/SVM-Fusion based land cover mapping and results integration
get_ensemble_label_rf+svm+svm48+vgg.py
Input：
- beijing_vgg128_48f.txt
- beijing_vgg128_prob.txt
- rf-model-origin24.save，svm-model-origin24.save，svm-vgg-48-sub128.save
- scaler_24f.save, scaler_48f.save
Output：
- beijing_svm24_rf24_svm48_vgg_pred.txt


Converting the predicted labels to final land cover maps
map_show.py
Input：
- beijing_svm24_rf24_svm48_vgg_pred.txt
- Landsat影像
Output：
- map_beijing_svm24_rf24_svm48_vgg_pred.tiff
