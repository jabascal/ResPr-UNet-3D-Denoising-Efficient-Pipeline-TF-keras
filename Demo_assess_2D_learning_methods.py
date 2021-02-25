# -*- coding: utf-8 -*-
"""
ResPr-UNet-3D-Denoising-Efficient-Pipeline-TF-keras
This repository contains data, code and results for the paper, A residual U-Net network with image prior for 3D image denoising, Proc. Eur. Signal Process. Conf. EUSIPCO, pp. 1264-1268, 2020. (hal-02500664)  
   
Demo_assess_learning_methods.py: Demo to assess and compare trained 2D CNN models  
    
Data: 
	Data provided in this repository consist of human thorax CT scans 
	(kits19 dataset, https://kits19.grand-challenge.org/) 
	corrupted with additive Gaussian noise. 

Requirements: 
    Tested on Python 3.7 and Tensorflow 2.4.1

If you use this code, please cite the following publication: 
    
    JFPJ Abascal et al. A residual U-Net network with image prior for 3D 
    image denoising, European Signal Processing Conference, 2020
    https://hal.archives-ouvertes.fr/hal-02500664
 
If you need to contact the author, please do so at juanabascal78@gmail.com

JFPJ Abascal
CREATIS, Biomedical Imaging Research Lab
CNRS UMR 5220
France    
    
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
# -------------------------------------------------------
def display_grid(plot_data, plot_titles, figsize=(6,6), nameSave = []):
    num_imgs = len(plot_data)
    num_cols = int(np.ceil(np.sqrt(num_imgs)))
    fig = plt.figure(figsize=(6,6))
    for i in range(num_imgs):
        # Start next subplot.
        plt.subplot(num_cols, num_cols, i + 1, title=plot_titles[i])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(plot_data[i], cmap='gray')
    if nameSave:
        fig.savefig(nameSave, dpi = 300, bbox_inches='tight') 
    return fig

# -------------------------------------------------------
# IMAGE PARAMETERS
img_width, img_height, img_channels = (256,256,1)
input_shape     = (img_width, img_height, img_channels)
perc_noise      = 0.05 # Additive Gaussian noise

# Data augmentation
#
# central crop before applying random crop
central_fraction = 0.8   
# -------------------------------------------------------
# PATHS, FILES
path_main = os.getcwd()
path_data_main = os.path.join(path_main,'Data')
name_data_test = 'Kits19_test_data_2subj_2pcnoise_crop_rnd.npz'

# Models
# name_models = ['kits19_200subj_data2k_Convnet_64_noisepc2_model_best.h5',
#                'kits19_200subj_data2k_UNet_32_noisepc2_model_best.h5',
#                'kits19_200subj_data2k_UNet_64_noisepc2_model_last.h5',
#                'kits19_200subj_data10k_UNet_32_noisepc2_model_best.h5',
#                'kits19_6subj_data2k_UNet_32_noisepc2_model_best.h5']
name_models = ['kits19_200subj_data2k_Convnet_64_noisepc2_model_best.h5',
               'kits19_200subj_data2k_UNet_32_noisepc2_model_best.h5']

#name_models_title = ['CNN', 'UN32', 'UN64','UN32 10k','UN32 6sub']
name_models_title = ['CNN', 'UN32']

name_save_main = 'kits19_UNet_ConvNet_comp2'
# -------------------------------------------------------
# LOAD TEST DATA     
data = np.load(os.path.join(path_data_main, name_data_test))
data_test = data['data_test']
data_test_noisy = data['data_test_noisy']

# Path for results (trained model and tensorboard)
path_results    = os.path.join(path_main, "Results")
if os.path.exists(path_results) is False:
    os.mkdir(path_results)
    print('Subdirectory created for results: ' + path_results)

# -------------------------------------------------------
# LOAD MODEL AND ASSESS
data_pred_err_all = []
data_pred_ssim_all = []
data_pred_all = []
for i, name_model in enumerate(name_models):
    # Load and predict
    model = keras.models.load_model(os.path.join(path_results, name_model))
    data_pred = model.predict(data_test_noisy)
    data_pred_all.append(data_pred)
    
    # Percentage MSE
    data_pred_err = np.mean(100*np.linalg.norm(data_test[:,:,:,0]-
                    data_pred[:,:,:,0], axis=(1, 2))/np.linalg.norm(data_test[:,:,:,0], axis=(1, 2)))
    data_pred_err_all.append(data_pred_err)
    
    # SSIM
    data_pred_ssim = tf.image.ssim(data_test[:,:,:,0],data_pred[:,:,:,0],max_val=1).numpy() 
    data_pred_ssim_all.append(data_pred_ssim)

    print('Perc. MSE: %.2f, SSIM: %.4f' % (data_pred_err, data_pred_ssim))


# Display result
ind = 2
num_models = len(data_pred_all)
plot_data = [data_test[ind,:,:,0], 
             data_test_noisy[ind,:,:,0]]+[data_pred_all[i][ind,:,:,0] 
                                          for i in range(num_models)] 
plot_titles = ['Raw image', 'Noisy image']+[name_models_title[i]+', err: %.2f' % (data_pred_err_all[i]) 
                                            for i in range(num_models)] 
fig = display_grid(plot_data, plot_titles,        
               figsize=(6,6),
               nameSave=os.path.join(path_results,
                             name_save_main+'_test_ex'+str(i)+'.png'))

# Zoom
indx, indy = np.arange(100,200), np.arange(150,250)
plot_data = [np.squeeze(data_test[np.ix_([ind],indx,indy,[0])]), 
             np.squeeze(data_test_noisy[np.ix_([ind],indx,indy,[0])])]+[
                 np.squeeze(data_pred_all[i][np.ix_([ind],indx,indy,[0])]) 
                                          for i in range(num_models)] 
fig = display_grid(plot_data[0:], plot_titles[0:],        
               figsize=(6,6),
               nameSave=os.path.join(path_results,
                             name_save_main+'_zoom'+'_test_ex'+str(i)+'.png'))

# Metrics
metrics = pd.DataFrame({'Perc. MSE': data_pred_err_all, 'SSIM': data_pred_ssim_all})
metrics.index = name_models_title
print(metrics)
metrics.plot.bar()

