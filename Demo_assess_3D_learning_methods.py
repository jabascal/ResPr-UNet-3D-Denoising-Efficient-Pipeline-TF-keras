# -*- coding: utf-8 -*-
"""
ResPr-UNet-3D-Denoising-Efficient-Pipeline-TF-keras
This repository contains data, code and results for the paper, 
A residual U-Net network with image prior for 3D image denoising, 
Proc. Eur. Signal Process. Conf. EUSIPCO, pp. 1264-1268, 2020. (hal-02500664)  
   
Demo_assess_3D_learning_methods.py: Demo to assess and compare trained 
3D CNN models  
    
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
        plt.imshow(plot_data[i], cmap='gray', vmin=0, vmax=1)
    if nameSave:
        fig.savefig(nameSave, dpi = 300, bbox_inches='tight') 
    return fig

# -------------------------------------------------------
# IMAGE PARAMETERS
input_shape = (32, 256, 256, 1)
img_depth, img_width, img_height, img_channels = input_shape
perc_noise      = 0.05 # Additive Gaussian noise

# Data augmentation
#
# central crop before applying random crop
central_fraction = 0.8   
# -------------------------------------------------------
# PATHS, FILES
path_main = os.getcwd()
# 
path_data_main = os.path.join(path_main,'Data')

name_data_test = "Kits19_256x_3D_test_data_5pcnoise_crop_rnd.npz"

# Models
name_models = [
            'kits19_200subj_256x256_3DUNet_32_noisepc5_model_best.h5',
            'kits19_200subj_data2k_Convnet_64_noisepc5_model_best.h5',
            'kits19_200subj_data2k_UNet_32_noisepc5_model_best.h5'
              ]

name_models_title = ['3DU-Net', '2DCNN', '2DU-Net']

path_models = 3*[os.path.join(path_main,'Results')]

model_types = ['3D']+2*['2D']

name_save_main = 'kits19_5pcnoise3D_256x_3D_vs_2D_test'

# Customs objects to load the model
custom_objects = [None for i in range(len(name_models))]

# -------------------------------------------------------
# LOAD TEST DATA     
data = np.load(os.path.join(path_data_main, name_data_test))
data_test = data['data_test']
data_test_noisy = data['data_test_noisy']

# Path for results (trained model and tensorboard)
path_results    = os.path.join(path_main, "Results\\Results_3D_den")
if os.path.exists(path_results) is False:
    os.mkdir(path_results)
    print('Subdirectory created for results: ' + path_results)

# -------------------------------------------------------
# LOAD MODEL AND ASSESS
data_pred_err_all = []
data_pred_ssim_all = []
data_pred_all = []
ind_z = 0 # index for displaying results
for i, (name_model, path_model, model_type) in enumerate(zip(name_models,path_models,model_types)):
    # Load and predict    
    model = keras.models.load_model(os.path.join(path_model, name_model))
    
    if model_type == '2D':
        data_noisy_this = data_test_noisy[:,ind_z,:,:,:]
    else:
        data_noisy_this = data_test_noisy
    
    data_test_this = data_test[:,ind_z,:,:,:]
        
    data_pred = model.predict(data_noisy_this)
    data_pred_all.append(data_pred)
    
    if model_type == '3D':
        data_pred = data_pred[:,ind_z,:,:,:]
        
    # Percentage MSE
    data_tets_norm = np.sqrt(np.sum(np.square(data_test)))
    data_pred_err = 100*np.sqrt(np.sum(np.square(data_test_this-data_pred)))/data_tets_norm
    data_pred_err_all.append(data_pred_err)
    
    # SSIM
    data_pred_ssim = np.mean(tf.image.ssim(data_test_this,data_pred,max_val=1)) 
    data_pred_ssim_all.append(data_pred_ssim)

    print('Perc. MSE: %.2f, SSIM: %.4f' % (data_pred_err, data_pred_ssim))


# Display result
#plt.imshow(data_test[7,10,:,:,0], cmap="gray");
plot_data = []
# Indices for dispalying results
ind_batch = 0; #2, 7, 1

num_models = len(data_pred_all)
plot_data = [data_test[ind_batch,ind_z,:,:,0],data_test_noisy[ind_batch,ind_z,:,:,0]]
for i, model_type in enumerate(model_types):
    if model_type == '2D':    
        plot_data += [data_pred_all[i][ind_batch,:,:,0]]
    elif model_type == '3D':                                           
        plot_data += [data_pred_all[i][ind_batch,ind_z,:,:,0]]
                                                     
plot_titles = ['Raw image', 'Noisy image']+[
    name_models_title[i]+', err: %.2f' % (data_pred_err_all[i]) 
                                            for i in range(num_models)] 
fig = display_grid(plot_data, plot_titles,        
               figsize=(6,6),
               nameSave=os.path.join(path_results,
                             name_save_main+'_test_ex'+str(ind_batch)+'.png'))

# Metrics
metrics = pd.DataFrame({'Perc. MSE': data_pred_err_all, 'SSIM': data_pred_ssim_all})
metrics.index = name_models_title
print(metrics)
metrics.plot.bar()

