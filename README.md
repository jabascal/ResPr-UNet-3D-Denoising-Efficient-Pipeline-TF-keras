# ResPr-UNet-3D-Denoising-Efficient-Pipeline-TF-keras
This repository contains data, code and results for the paper **A residual U-Net network with image prior for 3D image denoising, Proc. Eur. Signal Process. Conf. EUSIPCO, pp. 1264-1268, 2020. (hal-02500664)** 

We propose a modified U-Net architecture (ResPrU-Net) that exploits a prior image for 3D image denoising. The prior image is concatenated with the input and is connected to the output with a residual connection, as shown in the image below

![](https://github.com/jabascal/ResPr-UNet-3D-Denoising-Efficient-Pipeline-TF-keras/blob/main/ResPrUNet_sketch.jpg)

The idea behind ResPrU-Net is based a previously proposed variational method SPADE (JFPJ Abascal et al, A sparse and prior based method for 3D image denoising, Proc. Eur. Signal Process. Conf. EUSIPCO, 2019, hal-02056591). In this work, we extend the idea of exploiting a prior image to learning approaches. SPADE -- a modification of TV method by including a functional that promotes sparsity wrt to the prior image -- 
was shown to improve convergence and maintain image texture of the prior image, being less sensitive to noise and artefacts. ResPrU-Net provided better results than U-Net for high noise. 

In this repository, we provide and compare the proposed method to 2D and 3D variational and deep learning approaches. 

Learning approaches are trained using efficient tensorflow pipelines for image datasets, based on keras and tf.data. 

## Data 
Data provided in this repository consist of human thorax CT scans 
(kits19 dataset, https://kits19.grand-challenge.org/) 
corrupted with additive Gaussian noise. 

![](https://github.com/jabascal/ResPr-UNet-3D-Denoising-Efficient-Pipeline-TF-keras/blob/main/kit19_example.jpg) 

A subsample (8 subjects) of these data can be directly dowloaded from https://www.dropbox.com/sh/3sozaz9m7pyxcsa/AAAx1VULqc3T4rYC55vTiuJta?dl=0 (in png format). 

Realistic synthetic and experimental spectral CT knee data used in the publication will be provided later on.  

![](https://github.com/jabascal/ResPr-UNet-3D-Denoising-Efficient-Pipeline-TF-keras/blob/main/knee_example.jpg)

## Code
We provide Python code under tensorflow with keras. The python code has been tested  under Python 3.7 and tensorflow 2.4.1. 

## Summary of results ##
In progress ...

##  Repository files ##

The repository contains the following files:

### Python functions ###

- **Train_model_denoising2D.py:** Demo to train several 2D CNN models (simple CNN, U-Net) 

- **Train_model_denoising3D.py:** Under construction (Demo to train 3D models (3D U-Net, ResPr-UNet)

- **Demo_assess_learning_methods.m:** Under construction 

- **Demo_assess_variational_methods.m:** Under construction (Demo to assess TV, SPADE, among others)

- **Trained models and others:** Under construction 

## Data API ##
Tensorflow data API automatizes the data pipeline, chaining transformations (preprocessing and data augmentation), shuffling data. While the model is being trained for a given batch on the GPU, data are being prepared to be ready for next batch on the GPU.The current implementation is based on the following sources:  
-Aurélien Géron, Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition, O'Reilly Media, Inc.
ISBN: 9781492032649

-https://www.tensorflow.org/api_docs/python/tf/data/Dataset

-https://medium.com/deep-learning-with-keras/tf-data-build-efficient-tensorflow-input-pipelines-for-image-datasets-47010d2e4330

-https://keras.io/examples/vision/super_resolution_sub_pixel/

-https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/

If you use this code, please reference any of the following two publications: A residual U-Net network with image prior for 3D image denoising, Proc. Eur. Signal Process. Conf. EUSIPCO, pp. 1264-1268, 2020.  JFPJ Abascal et al, Material Decomposition in Spectral CT Using Deep Learning: A Sim2Real Transfer Approach, IEEE Access, vol. 9, pp. 25632-25647, 2021. 

If you need to contact the author, please do so at juanabascal78@gmail.com

