# -*- coding: utf-8 -*-
"""
ResPr-UNet-3D-Denoising-Efficient-Pipeline-TF-keras
This repository contains data, code and results for the paper, 
A residual U-Net network with image prior for 3D image denoising, 
Proc. Eur. Signal Process. Conf. EUSIPCO, pp. 1264-1268, 2020. (hal-02500664)  
   
Train_model_denoising2D.py: Demo to train several 2D CNN models (simple CNN, 
U-Net, ResNet). 

Learning approaches are trained using efficient tensorflow pipelines 
for image datasets, based on keras and tf.data. Using .chache (keep images in 
memory) and .prefetch (fetch and prepare data for next iteration), iterations 
are 1.6 times faster, for training done on 40,000 images. Data (image slices) 
are fetch from given directories.

Tensorboard callbacks have been used to visualize losses and denoised images 
during training. 
    
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
import pandas as pd
import numpy as np

import PIL
import PIL.Image
import pathlib
import time
import tensorflow as tf
#import tensorflow_datasets as tdfs
from tensorflow import keras
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import io
import random
# -------------------------------------------------------
# USER and PATHS
platform        = "win32" # "linux", "linux2"

# Select main path of the repository
# os.chdir(repo_path)

# Data paths for train and test
path_main = os.getcwd()
path_data_main   = os.path.join(path_main,'Data')
#subpath_data_train = "Train"
#subpath_data_test = "Test"
subpath_data_train = "Train"
subpath_data_test = "Test"

os.chdir(path_data_main)

# Path for results (trained model and tensorboard)
path_results    = os.path.join(path_main, "Results")
if os.path.exists(path_results) is False:
    os.mkdir(path_results)
    print('Subdirectory created for results: ' + path_results)

# Name specifications
name_save       = 'kits19_200subj_data2k_256x256'
#name_save       = 'kits19_6subj_data2k'

# Callback to continue training a model from checkpoint
mode_continue_training = False
if mode_continue_training == True:
    path_previous_model = os.path.join(path_results,
                            "kits19_200subj_data2k_UNet_32_noisepc5_model_best.h5")    
    name_save = name_save + '_cont300it'
# -------------------------------------------------------
# CNN
conv_model      = "UNet" # choose model: UNet, Convnet, ResNet
#
# Param for UNet and Convnet
conv_filt       = 32        # filters on first layer
kernel_size     = 3
activation      = "relu"
padding         = "same"
pool_size       = (2, 2)

# TRAIN
batch_size      = 64 # 4
epochs          = 20       # Early stopping on, check callbacks
optimizer       = "Adam"
learning_rate   = 1e-4      
validation_split = 0.1
train_eval_ratio = 0.9

model_name      = conv_model + '_' + str(conv_filt )
name_save       = name_save + '_' + model_name

period_check_point = 1

# IMAGE
img_width, img_height, img_channels = (256,256,1) # (64,64,1)
input_shape     = (img_width, img_height, img_channels)
mode_limited_data = True
if mode_limited_data == True:
    ds_train_size   = 2000 # number slices in training set
ds_test_size    = 100
# perc_noise      = 0.02 # Additive Gaussian noise
perc_noise      = 0.05 # Additive Gaussian noise

name_save = name_save + '_noisepc' + str(int(100*perc_noise))

# Data augmentation
#
# central crop before applying random crop
central_fraction = 0.7   

# Default display settings
plt.rcParams['figure.figsize'] = (4.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['font.size'] = 12
#plt.rcParams['lines.linestyle'] = '--'

# -------------------------------------------------------
# FUNCTIONS DECLARATION
#
# DATA IMAGE PROCESSING
# Data Augmentation
def flip(x: tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x

def random_crop(img):
    img = tf.image.central_crop(
        img, central_fraction)
    img = tf.image.random_crop(
            img, size=[img_height, img_width, 1])

    return img

# Additive noise
def add_gaussian_noise(img):
    #img_noisy = img + perc_noise*np.random.normal(0,1,img.shape)   
    img_noisy = img + perc_noise*tf.random.normal(img.shape,mean=0,stddev=1)
    return img_noisy
    
# Load and process image
def process_img(img):
    img = tf.image.decode_png(img, img_channels)
    
    # convert unit8 tensor to floats in range [0,1]
    img = tf.image.convert_image_dtype(img, tf.float32)
    #img = tf.image.resize(img, [img_width, img_height])
    return img 

def get_process_target(file_path: tf.Tensor):
    img = tf.io.read_file(file_path)
    img = process_img(img)
    return img 

#def get_process_noisy(file_path: tf.Tensor):
#    img = tf.io.read_file(file_path)
#    img = process_img(img)
#    img_noisy = add_gaussian_noise(img) 
#    return img_noisy

# LOG DIRECTORY FOR TENSORBOARD
def get_run_logdir(path_log):
    now         = time.strftime("run_%Y-%m-%d-%H-%M-%S")
    if platform == "linux" or platform == "linux2":
        # linux
        log_dir      = "{}//run-{}//".format(path_log, now, now)
    elif platform == "win32":
        # Windows
        log_dir      = "{}\\run-{}\\".format(path_log, now, now)
    return log_dir

# MODEL
#
# Simple ConvNet
def get_model_simple_convnet(input_shape = [28,28,1], 
                              conv_filt=64, 
                              kernel_size=3, 
                              activation="relu",
                              padding="same"):    
    conv_args = {"activation": activation, 
                 "padding": padding,
                 "kernel_size": kernel_size}
    model = keras.models.Sequential([            
        #Input(shape = input_shape),
        Conv2D(filters=conv_filt, input_shape=input_shape, **conv_args),
        Conv2D(filters=conv_filt, **conv_args),
        Conv2D(filters=1, kernel_size=1, activation=None)
    ])
    """  
    # Alternative definition
    model = keras.models.Sequential()
    model.add(Input(shape = input_shape))
    model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
    model.add(Conv2D(filters=1, kernel_size=1, activation="linear"))
    """
    """
    # Alternative definition
    inputs = Input(shape=input_shape)
    x = Conv2D(filters=64, **conv_args)(inputs)
    x = Conv2D(filters=64, **conv_args)(x)
    outputs = Conv2D(filters=1, kernel_size=1, **conv_args)(x)
    model = keras.Model(inputs, outputs)
    """    
    return model

# Simple 2D U-Net
def get_model_unet(input_shape = [28,28,1], 
                             conv_filt=32, 
                             kernel_size=3, 
                             activation="relu",
                             padding="same",
                             pool_size=pool_size):
    conv_args = {"activation": activation, 
                 "padding": padding,
                 "kernel_size": kernel_size}
    input_ = Input(shape=input_shape)
    conv1 = Conv2D(filters=conv_filt, **conv_args)(input_)
    conv2 = Conv2D(filters=conv_filt, **conv_args)(conv1)
    pool1 = MaxPool2D(pool_size=pool_size)(conv2)
    #
    conv3 = Conv2D(filters=2*conv_filt, **conv_args)(pool1)
    conv4 = Conv2D(filters=2*conv_filt, **conv_args)(conv3)
    pool2 = MaxPool2D(pool_size=pool_size)(conv4)
    #
    conv5 = Conv2D(filters=4*conv_filt, **conv_args)(pool2)
    conv6 = Conv2D(filters=2*conv_filt, **conv_args)(conv5)
    up1 = UpSampling2D(size=(2,2))(conv6)
    #
    conc1 = Concatenate()([conv4, up1])
    #
    conv7 = Conv2D(filters=2*conv_filt, **conv_args)(conc1)
    conv8 = Conv2D(filters=conv_filt, **conv_args)(conv7)
    up2 = UpSampling2D(size=(2,2))(conv8)
    #
    conc2 = Concatenate()([conv2, up2])
    #
    conv9 = Conv2D(filters=conv_filt, **conv_args)(conc2)
    conv10 = Conv2D(filters=conv_filt, **conv_args)(conv9)
    #
    output = Conv2D(filters=1, kernel_size=1, activation=None)(conv10)
    #
    model = keras.Model(inputs=[input_], outputs=[output])
    return model

# RESNET
#
# Residual unit
# Implementation based on (Aurelien Geron, Hands-on Machine Learning 
# with Scikit-learn, Keras & Tensorflow)
class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, features_change=False, activation="relu", **kwargs):
        super(ResidualUnit, self).__init__(**kwargs)
        self.activation = keras.activations.get(activation)
    # def build(self, filters): 
        self.main_layers = [
            Conv2D(filters, 3, strides=1,
                                padding="same", use_bias=False),
            BatchNormalization(),
            self.activation,
            Conv2D(filters, 3, strides=1, 
                                padding="same", use_bias=False),
            BatchNormalization()]
        self.skip_layers = []
        if features_change is True:
            self.skip_layers = [
                Conv2D(filters, 1, strides=1, 
                       padding="same", use_bias=False),
                BatchNormalization()]
            
    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)        
        return self.activation(Z + skip_Z)
    
    def get_config(self):
        # base_config = super().get_config()
        # direct loading model currently not working
        base_config = super(ResidualUnit, self).get_config()
        return {**base_config, 
                "activation": keras.activations.serialize(self.activation)}
#                "ResidualUnit": self.main_layers}
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
# layer = ResidualUnit(filters=64)
# print(layer.get_config())  
    
def get_model_resnet(input_shape = [28,28,1], filters = 64):
# Implementation based on (Aurelien Geron, Hands-on Machine Learning 
# with Scikit-learn, Keras & Tensorflow)
    # Made strides=1 to keep same size (and nott pooling either) , 
    # remove flattten/dense layers, and added a Conv2 at the end
    model = keras.models.Sequential()
    model.add(Conv2D(filters, 3, strides=1, input_shape=input_shape,
                     padding="same", use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    prev_filters = filters   
    for filters in [filters]*3 + [2*filters]*4: # + [4*filters]*6 + [6*filters]*3:
        features_change = False if filters == prev_filters else True
        model.add(ResidualUnit(filters=filters, features_change=features_change))
        prev_filters = filters
    model.add(Conv2D(filters=1, kernel_size=1, padding="same",
                     activation=None,use_bias=False))        
    return model 


# IMAGE VISUALIZATION
def displaySlices(X_Tensor, titles, figsize=(12,4), nameSave = []):
    # displaySlices(X_Tensor, titles)
    # displaySlices(X_Tensor, titles, figsize=(12,6), nameSave = 'name.png')
    #
    # Display slices (target, noisy image and prior)
    #
    # Inputs:
    #
    # X_Tensor = tuple of 2D arrays. 
    # X_Tensor =  (X_batch_train[0,:,:,0],S_batch_train[0,:,:,0],S_batch_pr_train[0,:,:,0])
    # titles    = ('Target','Noisy','Prior') 
    num_dim    = len(X_Tensor)
    fig, axarr = plt.subplots(1, num_dim, figsize=figsize)
    fig.tight_layout()    
    for i in range(num_dim):
        img0 = axarr[i].imshow(X_Tensor[i],cmap='gray')
        divider = make_axes_locatable(axarr[i])
        cax = divider.append_axes("right", size="7%", pad=0.05)
        plt.colorbar(img0, cax = cax) 
        axarr[i].set_title(titles[i])
        axarr[i].axis('off')
    plt.show()
    if nameSave:
        fig.savefig(nameSave, dpi = 300, bbox_inches='tight') 
    return fig

def image_grid(images_eval_rnp, images_eval_rnp_labels):
  """Return a grid image as a matplotlib figure."""
  # Create a figure to contain the plot.

  figure = plt.figure(figsize=(12,4))
  for i in range(3):
    # Start next subplot.
    plt.subplot(1, 3, i + 1, title=images_eval_rnp_labels[i])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images_eval_rnp[i], cmap=plt.cm.binary)

  return figure

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call.
  From https://www.tensorflow.org/tensorboard/image_summaries
  """
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

def log_denoised_image(epoch, logs):
    images_eval_pred = model.predict(images_eval_noisy)
    images_eval_rnp = [images_eval[0,:,:,:], 
                        images_eval_noisy[0,:,:,:],
                        images_eval_pred[0,:,:,:]]
    images_eval_rnp_labels = ['Raw', 'Noisy', 'Denoised']

    figure = image_grid(images_eval_rnp, images_eval_rnp_labels)    
    
    with file_writer_cm.as_default():
        tf.summary.image("Denoised image", plot_to_image(figure), step=epoch)                     

def get_imgs_from_dataset(ds_test, ds_test_size):
    # Take images from data set ds_test: (data_test, data_test_noisy) 
    data_test = []
    data_test_noisy = []
    count = 0
    for a_n, a in ds_test.take(ds_test_size):
        data_test_this = a.numpy()  
        data_test_noisy_this = a_n.numpy() 
        if count == 0:
            data_test = data_test_this
            data_test_noisy = data_test_this
            count = 1
        else:            
            data_test = np.append(data_test, data_test_this, axis=2)
            data_test_noisy = np.append(data_test_noisy, data_test_noisy_this, axis=2)
    data_test = np.transpose(data_test, (2,0,1))
    data_test_noisy = np.transpose(data_test_noisy, (2,0,1))
    data_test = data_test[:,:,:,np.newaxis] 
    data_test_noisy = data_test_noisy[:,:,:,np.newaxis] 
    return data_test, data_test_noisy
# -------------------------------------------------------
# CALLBACKS 
# Tensorboard
# tensorboard --logdir=log_dir
log_dir = get_run_logdir(path_results)    
tensorboard_cb = keras.callbacks.TensorBoard(
    log_dir,
    histogram_freq=10, # Histograms of weights
    write_grads=True,  # Gradients values
    write_images=True, # Images for weights
    profile_batch = '5,10') # Profiling: improve performance https://www.tensorflow.org/guide/profiler

# save best model every few iterations
checkpoint_best_cb = keras.callbacks.ModelCheckpoint(
    os.path.join(path_results, name_save+'_model_best.h5'), 
    period=period_check_point, 
    save_best_only=True,
    monitor='val_mse',)

# early stopping based on a given metric
# earlystop_cb = keras.callbacks.EarlyStopping(
#     os.path.join(path_results, name_save+'_model_best.h5')
#     patience=period_check_point, 
#     monitor='val_mse',
#     restore_best_weights=True)

# profiler API to improve data pipeline efficiency
# profile_batch indicate batches on which apply profiler
# profile_cb = tf.keras.callbacks.TensorBoard(
#     log_dir)#,
    #profile_batch='20, 40')

# Predict (Denoised) an image every epoch to visualize in tensorboard
file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')
image_den_cb = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_denoised_image)

callbacks = [tensorboard_cb, checkpoint_best_cb, image_den_cb]

# -------------------------------------------------------
# DATASETS

# Create file list 
# 
# Train set
# tf.data.Dataset.from_tensor_slices with glob more efficient
data_dir_train = pathlib.Path(subpath_data_train) 
filenames_train = list(data_dir_train.glob('*/*.png'))
fnames_train = [str(fname) for fname in filenames_train]
print(str(filenames_train[0]))
PIL.Image.open(str(filenames_train[0]))

   
# Test set
data_dir_test = pathlib.Path(subpath_data_test) 
filenames_test = list(data_dir_test.glob('*/*.png'))
fnames_test = [str(fname) for fname in filenames_test]

# Datasets ds train and test
if mode_limited_data == False:
    ds_train_size  = len(fnames_train)    
else:
    # Randomly select slices from the entire training set
    random.shuffle(fnames_train)

filelist_train_ds = tf.data.Dataset.from_tensor_slices(fnames_train[:ds_train_size]) 

# Check train ds: display images (can simply iterate over items in the data set)
# filelist_ds     = tf.data.Dataset.list_files(str(data_dir/'*/*'))
for a in filelist_train_ds.take(3):
    fname = a.numpy().decode("utf-8") # not in TF 1.4
    print(fname)
    display(PIL.Image.open(fname))

# Split train and eval 
ds_train = filelist_train_ds.take(int(ds_train_size*train_eval_ratio))
ds_eval = filelist_train_ds.skip(int(ds_train_size*train_eval_ratio))

print("Train data num: ", ds_train.cardinality().numpy())
print("Eval data num: ", ds_eval.cardinality().numpy())

# Test set
filelist_test_ds = tf.data.Dataset.from_tensor_slices(fnames_test[:ds_test_size])
ds_test = filelist_test_ds.take(ds_test_size)
print("Test data num: ", ds_test.cardinality().numpy())

# Transform data
# Load, normalize, data augmentation (random crop, horizontal flip), 
# corrupt with additive Gaussian noise 
map_ops     = [get_process_target, flip, random_crop] # Select data aug ops (flip, random_crop)

# Shard operator (for several workers) used early in the dataset pipeline (when reading from a set of TFRecord files)
# ds_train = ds_train.shard()

# TRAIN
for f in map_ops:
    ds_train = ds_train.map(lambda x: f(x),
        num_parallel_calls=tf.data.AUTOTUNE) # num_parallel_calls for multithreading   
"""
for a in ds_train.take(2):
    plt.imshow(a.numpy()[:,:,0])
    plt.show() 
"""

ds_train = ds_train.map(  # return tuple (noisy, target) for training
        lambda x: (add_gaussian_noise(x), x),
        num_parallel_calls = tf.data.AUTOTUNE)
print('Examples of training data random crops')
for a,b in ds_train.take(5):
    displaySlices([a.numpy()[:,:,0],b.numpy()[:,:,0] ], 
               ['Noisy image', 'Raw image'], figsize=(12,4))   
# EVAL
for f in map_ops:
    ds_eval = ds_eval.map(lambda x: f(x),
        num_parallel_calls=tf.data.AUTOTUNE)   
ds_eval = ds_eval.map(  
        lambda x: (add_gaussian_noise(x), x),
        num_parallel_calls = tf.data.AUTOTUNE)

# TEST    
for f in map_ops:
    ds_test = ds_test.map(lambda x: f(x),
        num_parallel_calls=tf.data.AUTOTUNE)   
ds_test = ds_test.map(  
        lambda x: (add_gaussian_noise(x), x),
        num_parallel_calls = tf.data.AUTOTUNE)


# Test data
data_test, data_test_noisy = get_imgs_from_dataset(ds_test, ds_test_size)

"""
# Save test subset
np.savez(os.path.join(path_data_main,
                            name_save + '_test_crop_rnd' + '.npz'), 
                          data_test = data_test, data_test_noisy=data_test_noisy)    

# Save train subset of data
data_train, data_train_noisy = get_imgs_from_dataset(ds_train, ds_train_size)

np.savez(os.path.join(path_data_main,
                           name_save + '_train_crop_rnd' + '.npz'), 
                          data_train = data_train, data_train_noisy=data_train_noisy)    
"""

# Buffer size greater than or equal to the full size of the dataset 
buffer_size_train = ds_train.cardinality().numpy()
buffer_size_eval = ds_eval.cardinality().numpy()

# DS: .cache() keeps the images in memory after they're loaded off disk during 
# the first epoch. If dataset is too large (also use this method to create 
# a performant on-disk cache.
# .prefetch() overlaps data preprocessing and model execution while training
# To randomize iteration order, call shuffle after calling cache
# 
# Without .cache it is 23 % slower, and without .cache and .prefetch 64% slower
ds_train_batched = ds_train.batch(batch_size).cache()  # 
##ds_train_batched = ds_train_batched.repeat(3) # repeat items of original data set
ds_train_batched = ds_train_batched.shuffle(buffer_size=buffer_size_train, 
                                            reshuffle_each_iteration=True,
                                            seed = 50)
ds_train_batched = ds_train_batched.prefetch(tf.data.experimental.AUTOTUNE)  
# 
ds_eval_batched = ds_eval.batch(batch_size).cache()
##ds_eval_batched = ds_eval_batched.repeat(3)
ds_eval_batched = ds_eval_batched.shuffle(buffer_size=buffer_size_eval, 
                                          reshuffle_each_iteration=True,
                                            seed = 50)
ds_eval_batched = ds_eval_batched.prefetch(tf.data.experimental.AUTOTUNE)  

 
# Images eval for tensorboard summary
images_eval = []
images_eval_noisy = []
count = 0
for a_n, a in ds_eval_batched.take(1):
    images_eval = a.numpy()
    images_eval_noisy = a_n.numpy()

# Write images to tensorboard
# Create a file writer 
# log_dir = get_run_logdir(path_results)
# file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')
# images_eval_pred = model.predict(images_eval_noisy)

"""
# Log given image tensor
with file_writer_cm.as_default():
    N = images_eval.shape
    imgs = np.reshape([images_eval[0,:,:,0], images_eval[0,:,:,0]], 
                      (-1,N[1],N[2],1))
    tf.summary.image('Eval image', imgs, max_outputs=2, step=0)
"""
"""
# Log a general figure
images_eval_rnp = [images_eval[0,:,:,0], 
                   images_eval_noisy[0,:,:,0],
                   images_eval_pred[0,:,:,0]]
figure = image_grid()
with file_writer_cm.as_default():
    tf.summary.image("Denoised image", plot_to_image(figure), step=0)
    
"""

# -------------------------------------------------------
# MODEL AND TRAIN 
#
if mode_continue_training == True:
    # Continue training a model from checkpoint
    #
    # Load previous model
    model = keras.models.load_model(path_previous_model) 
    
else:
    # Define model
    if conv_model == "Convnet": # UNet, Convnet
        model   = get_model_simple_convnet(
            input_shape=input_shape, 
            conv_filt=conv_filt,
            kernel_size=kernel_size,
            activation=activation,
            padding=padding)
    elif conv_model == "UNet":    
        model   = get_model_unet(
            input_shape=input_shape, 
            conv_filt=conv_filt,
            kernel_size=kernel_size,
            activation=activation,
            padding=padding,
            pool_size=pool_size)
    elif conv_model == "ResNet": 
        model =  get_model_resnet(input_shape = input_shape, filters=conv_filt)

    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)    
    model.compile(optimizer=optimizer,
              loss=loss_fn, 
              metrics=["mse", 
                       "mae", # equivalent to mape for [0,1] images, mape not working properly
                       #"mean_absolute_percentage_error" # mape = tf.keras.losses.MeanAbsolutePercentageError()
                       ])         
         
model.summary()
# model.layers

# TRAIN
#
# Fit the model
history = model.fit(ds_train_batched, 
                    epochs=epochs, 
                    validation_data=ds_eval_batched, 
                    callbacks=callbacks)

# Losses
pd.DataFrame(history.history).plot(figsize=[8, 5], logy=True)
plt.show()
# -------------------------------------------------------
# PREDICT AND ERROR
# 
# Load best model
model = keras.models.load_model(os.path.join(path_results, 
                                              name_save+'_model_best.h5'))
                  
# Test data
data_test, data_test_noisy = get_imgs_from_dataset(ds_test, ds_test_size)

"""
# Save test data
np.savez(os.path.join(path_data_main,
                            'Kits19_test_data_2subj_2pcnoise_crop_rnd' + '.npz'), 
                          data_test = data_test, data_test_noisy=data_test_noisy)    
"""
# Predict (denoise image)
data_pred = model.predict(data_test_noisy)
data_pred_err = np.mean(100*np.linalg.norm(data_test[:,:,:,0]-
                data_pred[:,:,:,0], axis=(1, 2))/np.linalg.norm(data_test[:,:,:,0], axis=(1, 2)))
print('Error %.1f' % data_pred_err)

# Evaluation
#with tf.name_scope("eval"):    
#    mse_rel    = tf.norm(X - X_)/tf.norm(X)

# Display result
print('Example of restored images')
for i in range(5):
    displaySlices([data_test[i,:,:,0], data_test_noisy[i,:,:,0], 
                   data_pred[i,:,:,0]], 
               ['Raw image', 'Noisy image', 'Denoised image'], 
               figsize=(12,4),
               nameSave=os.path.join(path_results,
                            name_save+'_test_ex'+str(i)+'.png'))

# -------------------------------------------------------
# SAVE MODEL
model.save(os.path.join(path_results, name_save + "_model_last.h5"))
# model = keras.models.load_model('UNet_32.h5')
# weights, biases= history.model.layers[1].weights
# -------------------------------------------------------
