# ResPr-UNet-3D-Denoising-Efficient-Pipeline-TF-keras
This repository contains data, code and results for the paper **A residual U-Net network with image prior for 3D image denoising, Proc. Eur. Signal Process. Conf. EUSIPCO, pp. 1264-1268, 2020. (hal-02500664)** 

We propose a modified U-Net architecture (ResPrU-Net) that exploits a prior image for 3D image denoising. The prior image is concatenated with the input and is connected to the output with a residual connection, as shown in the image below

![](https://github.com/jabascal/ResPr-UNet-3D-Denoising-Efficient-Pipeline-TF-keras/blob/main/figures/ResPrUNet_sketch.jpg)

The idea behind ResPrU-Net is based a previously proposed variational method SPADE (JFPJ Abascal et al, A sparse and prior based method for 3D image denoising, Proc. Eur. Signal Process. Conf. EUSIPCO, 2019, hal-02056591). In this work, we extend the idea of exploiting a prior image to learning approaches. SPADE -- a modification of TV method by including a functional that promotes sparsity wrt to the prior image -- 
was shown to improve convergence and maintain image texture of the prior image, being less sensitive to noise and artefacts. ResPrU-Net provided better results than U-Net for high noise. 

In this repository, we provide and compare the proposed method to 2D and 3D variational and deep learning approaches. 

Learning approaches are trained using efficient tensorflow pipelines for image datasets, based on keras and tf.data (see below for more details). 

Tensorboard callbacks have been used to visualize losses, metrics, denoised images and histograms of the learned weights during training. TensorFlow Profiler is also used to profile the execution of the code. 

## Data 
Data provided in this repository consist of human thorax CT scans 
(kits19 dataset, https://kits19.grand-challenge.org/) 
corrupted with additive Gaussian noise. 

![](https://github.com/jabascal/ResPr-UNet-3D-Denoising-Efficient-Pipeline-TF-keras/blob/main/figures/kit19_example.jpg) 

A subsample of kits19 data can be directly dowloaded from https://www.dropbox.com/sh/3sozaz9m7pyxcsa/AAAx1VULqc3T4rYC55vTiuJta?dl=0 (in png format, data with loosy conversion from float64 to unit8; fetch original data from the kits19 link): 
- **'\Data\Train\':** All slices for 6 subjects.  
- **'\Data\Test':** All slices for 2 subjects. 
- **'Kits19_train_data_200subj_2kimgs_2pcnoise_crop_rnd.npz\':** 2k slices randomly selectec from 200 subjects. 
- **'Kits19_test_data_2subj_2pcnoise_crop_rnd.npz\':** Data used for test corresponding to two subjects. 

Realistic synthetic and experimental spectral CT knee data used in the publication will be provided later on.  

![](https://github.com/jabascal/ResPr-UNet-3D-Denoising-Efficient-Pipeline-TF-keras/blob/main/figures/knee_example.jpg)

## Installation
We provide Python code under tensorflow with keras. The python code has been tested  under Python 3.8 and tensorflow 2.4.1.  

Different TF versions required specific cudatoolkit version. With anaconda one can handle several cuda versions in one machine. To install the tested version on Windows:  
   
```
conda create -n tf2 anaconda python=3.8 
conda install cudatoolkit=11.0 cudnn=8.0 -c=conda-forge
pip install --upgrade tensorflow-gpu==2.4.1 
python
import tensorflow as tf 
tf.test.is_gpu_available()
```

For TF compatibilities: https://www.tensorflow.org/install/source_windows 
For setting up Tensorflow-GPU with Cuda and Anaconda on Windows: 
https://towardsdatascience.com/setting-up-tensorflow-gpu-with-cuda-and-anaconda-onwindows-2ee9c39b5c44

## Summary of results ##

### 2D learning-based approaches ###

The following figures show results for U-Net (with 32 and 64 filters on the first layer; 332,641 parameters) and a simple 3-layer ConvNet (37,633 parameters), for 2 % additive Gaussian noise. Unless specified, training data is comprised of 2000 slices taking randomly from all 200 subjects. Similar results were obtained by training on 10000 (10k) slices. Worse results corresponded to considering only 6 subjects (6sub, ~2k slices). U-Net 32 trained on 1h30 min and the ConvNet in 28 min. 
Train data file: (Kits19_train_data_200subj_2kimgs_2pcnoise_crop_rnd.npz. Test data file: Kits19_test_data_2subj_2pcnoise_crop_rnd.npz.  

![](https://github.com/jabascal/ResPr-UNet-3D-Denoising-Efficient-Pipeline-TF-keras/blob/main/figures/kits19_UNet_ConvNet_comp_zoom_test_ex3_sm.png) 

The following figures show results for U-Net and a simple 3-layer ConvNet (trained on 2k slices), split Bregman TV and BM3D, for 5 % additive Gaussian noise. 

![](https://github.com/jabascal/ResPr-UNet-3D-Denoising-Efficient-Pipeline-TF-keras/blob/main/figures/kits19_5pcnoise_TV_BM3D_CNN_Unet.png) 

### 3D learning-based approaches ###

The following figures show results for 2D CNN and U-Net and 3D U-Net for 5 % additive Gaussian noise. 3D U-Net was trained on 200 subjects on randonmly cropped patches of 32x256x256, for 300 epochs (preliminary result; improved results should be obtained for further iterations).   

![](https://github.com/jabascal/ResPr-UNet-3D-Denoising-Efficient-Pipeline-TF-keras/blob/main/figures/kits19_5pcnoise3D_256x_3D_vs_2D_UNet_CNN_ex0.png.png) 

##  Repository files ##

The repository contains the following files:

### Python functions ###

- **Train_model_denoising2D.py:** Demo to train several 2D CNN models (simple CNN, simple ResNet, U-Net) 

- **Train_model_denoising3D.py:** (Demo to train 3D models (3D CNN, 3D U-Net, ResPr-UNet; ResPr-UNet under construction)

- **Demo_assess_2D_learning_methods.py:** Demo to compared the trained 2D models 

- **Demo_assess_3D_learning_methods.py:** Demo to compared the trained 3D models 

- **Demo_assess_variational_methods.py:** Under construction (Demo to assess TV, SPADE, among others)

- **Trained models and others:** Under construction 

### Trained models ###
- **kits19_200subj_data2k_Convnet_64_noisepc2_model_best.h5, kits19_200subj_data2k_Convnet_64_noisepc5_model_best.h5:** Simple ConvNet (3 layers, 64 filters) trained on 2% and 5% additive Gaussian noise for 2,000 slices randomly selected from 200 subjects. 

- **kits19_200subj_data2k_UNet_32_noisepc2_model_best.h5, kits19_200subj_data2k_UNet_32_noisepc5_model_best.h5:** U-Net (32 filters on the first layer) trained on 2% and 5% additive Gaussian noise for 2,000 slices randomly selected from 200 subjects. 

- **kits19_200subj_32x256x256_3DUNet_32_noisepc5_model_best.h5:** 3D U-Net (32 filters on the first layer) trained on 5% additive Gaussian noise, 200 subjects, randonmly cropped patches of 32x256x256, for 300 epochs. 

## Data API ##
Tensorflow data API automatizes the data pipeline, chaining transformations (preprocessing and data augmentation), shuffling data. While the model is being trained for a given batch on the GPU, data are being prepared to be ready for next batch on the GPU.
On the tests carried out here, using .chache (keep images in memory) and .prefetch (fetch and prepare data for next iteration), iterations were 1.6 times faster. Using tf datasets, data (image slices) are automatically fetched and pre-processed from the given directories.

The current implementation is based on the following sources:  
-Aurélien Géron, Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition, O'Reilly Media, Inc.
ISBN: 9781492032649

-https://www.tensorflow.org/api_docs/python/tf/data/Dataset

-https://medium.com/deep-learning-with-keras/tf-data-build-efficient-tensorflow-input-pipelines-for-image-datasets-47010d2e4330

-https://keras.io/examples/vision/super_resolution_sub_pixel/

-https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/

If you use this code, please reference any of the following two publications: A residual U-Net network with image prior for 3D image denoising, Proc. Eur. Signal Process. Conf. EUSIPCO, pp. 1264-1268, 2020.  JFPJ Abascal et al, Material Decomposition in Spectral CT Using Deep Learning: A Sim2Real Transfer Approach, IEEE Access, vol. 9, pp. 25632-25647, 2021. 

If you need to contact the author, please do so at juanabascal78@gmail.com

