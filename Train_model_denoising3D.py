# -*- coding: utf-8 -*-
"""
ResPr-UNet-3D-Denoising-Efficient-Pipeline-TF-keras
This repository contains data, code and results for the paper, 
A residual U-Net network with image prior for 3D image denoising, 
Proc. Eur. Signal Process. Conf. EUSIPCO, pp. 1264-1268, 2020. (hal-02500664)  
   
Train_model_denoising3D.py: Demo to train several 3D CNN models (simple CNN, 
U-Net, ResNet). 

Learning approaches are trained using efficient tensorflow pipelines 
for image datasets, based on keras and tf.data. 
    3D loading takes 92% time: 18s per epoch for 15 subjects (1.2 min per subject)
    Using chache (keep images in memory) before cropping images 
    reduces epoch to 6s, 20s for 75 subjects, 238s (230s with prefetch) for 200 subjects. 
    It takes 955s on 1st epoch for 200 subjects, 230s on 2nd epoch 
    using 70GB on RAM     

Tensorboard callbacks have been used to visualize losses and denoised images 
during training    
    
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
from tensorflow.keras import layers

import nibabel as nib

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import io
import random

# -------------------------------------------------------
# USER and PATHS
platform        = "win32" # "linux", "linux2"

# Select main path of the repository
path_main = os.getcwd()
# Select data path: subjects will be divided in training, evaluation and test
path_data_main   = os.path.join(path_main,'\kits19-interpolated')

# Path for results (trained model and tensorboard)
path_results    = os.path.join(path_main, "Results_3D_den")
if os.path.exists(path_results) is False:
    os.mkdir(path_results)
    print('Subdirectory created for results: ' + path_results)

    
# Name specifications
name_save       = 'kits19_50subj_256x256'

# Callback to continue training a model from checkpoint
mode_continue_training = False
if mode_continue_training == True:
    path_previous_model = os.path.join(path_results,
                            "kits19_200subj_256x256_3DUNet_32_noisepc5_model_best.h5")    
    name_save = name_save + '_cont320it'
# -------------------------------------------------------
# CNN
conv_model      = "3DUNet" # choose model: 3DUNet, 3DCNN, 3DPrResUNet
#
# Param for UNet and Convnet
conv_filt       = 32        # filters on first layer
kernel_size     = 3
activation      = "relu"
padding         = "same"
pool_size       = (2, 2, 2)

# TRAIN
batch_size      = 2 # 4
epochs          = 5000       # Early stopping on, check callbacks
optimizer       = "Adam"
learning_rate   = 1e-4      

model_name      = conv_model + '_' + str(conv_filt )
name_save       = name_save + '_' + model_name

period_check_point = 1

# IMAGE
# (channels, conv_dim1, conv_dim2, conv_dim3)
img_depth, img_width, img_height, img_channels = (32, 256, 256, 1) #
input_shape     = (img_depth, img_width, img_height, img_channels)

# Resize loading images to same size before cropping
img_shape_load = (100, 600, 600, 1)

# Split data
# validation_split = 0.1
# train_eval_ratio = 0.9
ds_test_size    = 1
ds_eval_size    = 4
ds_train_size   = 45

mode_limited_data = False
if mode_limited_data == True:
    ds_train_size   = 200 # number slices in training set

perc_noise      = 0.1 # Additive Gaussian noise

name_save = name_save + '_noisepc' + str(int(100*perc_noise))

# Data augmentation
#
# central crop before applying random crop
central_fraction = 0.8   

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
def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume   

def read_nifti_file(filepath):
    """Read and load volume:"""
    # Read file
    scan = np.asarray(nib.load(filepath).get_fdata())
    scan = normalize(scan) # do later af filter!
    scan = tf.convert_to_tensor(scan, tf.float32)        
    return scan

def resize_scan(scan):
    # Resize scans to equal size: Crop or padd with zeros
    scan = tf.transpose(scan, (1,2,0))
    scan = tf.image.resize_with_crop_or_pad(scan, 
                                                  img_shape_load[1],
                                                  img_shape_load[2])
    scan = tf.transpose(scan, (2,0,1))
    scan = tf.image.resize_with_crop_or_pad(scan, 
                                                  img_shape_load[0],
                                                  img_shape_load[2])
    scan = tf.expand_dims(scan, axis=3)
    return scan        

def gen(fnames):
    # Generator function (with yield) returns a generator object 
    # Generators are iterators that produces (yields) values pass to yield
    # g = gen(fnames=fnames_train)
    # img=next(g); plt.imshow(img[0,:,:,0]);
    for filepath in fnames:
        # Load image
        scan = read_nifti_file(filepath)
        # Resize
        scan = resize_scan(scan)
        # Filter
        #scan = normalize_ds(scan)
        yield scan

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

def random_crop_two_imgs(input_image, real_image):
    input_image = tf.image.central_crop(input_image, central_fraction)
    real_image = tf.image.central_crop(real_image, central_fraction)    
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_size = list(input_shape)
    cropped_size.insert(0, 2)
    cropped_image = tf.image.random_crop(stacked_image, 
                                         size=cropped_size)
    return cropped_image[0], cropped_image[1]

# Additive noise
def add_gaussian_noise(img):
    img_noisy = img + perc_noise*tf.random.normal(img.shape,mean=0,stddev=1)
    return img_noisy

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
# Simple 3D ConvNet
# input: batch_shape + (channels, conv_dim1, conv_dim2, conv_dim3)
def get_model_simple_convnet(input_shape = input_shape, 
                              conv_filt=64, 
                              kernel_size=3, 
                              activation="relu",
                              padding="same"):    
    conv_args = {"activation": activation, 
                 "padding": padding,
                 "kernel_size": kernel_size}
    
    inputs = layers.Input(shape = input_shape)
    x = layers.Conv3D(filters=conv_filt, **conv_args)(inputs)
    x = layers.Conv3D(filters=conv_filt, **conv_args)(x)
    outputs = layers.Conv3D(filters=1, kernel_size=1, activation=None)(x)        
    model = keras.Model(inputs, outputs) 
    return model

# Simple 3D U-Net
def get_model_unet(input_shape = input_shape, 
                             conv_filt=32, 
                             kernel_size=3, 
                             activation="relu",
                             padding="same",
                             pool_size=pool_size):
    conv_args = {"activation": activation, 
                 "padding": padding,
                 "kernel_size": kernel_size}
    inputs = layers.Input(shape=input_shape)
    conv1 = layers.Conv3D(filters=conv_filt, **conv_args)(inputs)
    conv2 = layers.Conv3D(filters=conv_filt, **conv_args)(conv1)
    pool1 = layers.MaxPooling3D(pool_size=pool_size)(conv2)
    #
    conv3 = layers.Conv3D(filters=2*conv_filt, **conv_args)(pool1)
    conv4 = layers.Conv3D(filters=2*conv_filt, **conv_args)(conv3)
    pool2 = layers.MaxPooling3D(pool_size=pool_size)(conv4)
    #
    conv5 = layers.Conv3D(filters=4*conv_filt, **conv_args)(pool2)
    conv6 = layers.Conv3D(filters=2*conv_filt, **conv_args)(conv5)
    up1 = layers.UpSampling3D(size=(2,2,2))(conv6)
    #
    conc1 = layers.Concatenate()([conv4, up1])
    #
    conv7 = layers.Conv3D(filters=2*conv_filt, **conv_args)(conc1)
    conv8 = layers.Conv3D(filters=conv_filt, **conv_args)(conv7)
    up2 = layers.UpSampling3D(size=(2,2,2))(conv8)
    #
    conc2 = layers.Concatenate()([conv2, up2])
    #
    conv9 = layers.Conv3D(filters=conv_filt, **conv_args)(conc2)
    conv10 = layers.Conv3D(filters=conv_filt, **conv_args)(conv9)
    #
    output = layers.Conv3D(filters=1, kernel_size=1, activation=None)(conv10)
    #
    model = keras.Model(inputs=[inputs], outputs=[output])
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
        img0 = axarr[i].imshow(X_Tensor[i],cmap='gray', vmin=0, vmax=1)
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
    images_eval_rnp = [images_eval[0,0,:,:,:], 
                        images_eval_noisy[0,0,:,:,:],
                        images_eval_pred[0,0,:,:,:]]
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
        data_test_this = data_test_this[np.newaxis,:,:,:]
        data_test_noisy_this = data_test_noisy_this[np.newaxis, :,:,:]
        if count == 0:
            data_test = data_test_this
            data_test_noisy = data_test_noisy_this
            count = 1
        else:            
            data_test = np.concatenate((data_test, data_test_this))
            data_test_noisy = np.concatenate((data_test_noisy, data_test_noisy_this))
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
    profile_batch = (2,3)) # Profiling: improve performance https://www.tensorflow.org/guide/profiler

# save best model every few iterations
checkpoint_best_cb = keras.callbacks.ModelCheckpoint(
    os.path.join(path_results, name_save+'_model_best.h5'), 
    period=period_check_point, 
    save_best_only=True,
    monitor='val_mse')

# Predict (Denoised) an image every epoch to visualize in tensorboard
file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')
image_den_cb = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_denoised_image)

callbacks = [tensorboard_cb, checkpoint_best_cb, image_den_cb]

# -------------------------------------------------------
# DATASETS

# Create file list 
# 
# Train set
data_dir_train = pathlib.Path(path_data_main) 
filenames_train = list(data_dir_train.glob('*/imaging.nii.gz'))
fnames_train = [str(fname) for fname in filenames_train]
img = read_nifti_file(filenames_train[0])
plt.imshow(img[10,:,:]); plt.colorbar(); plt.show();

# Datasets ds train and test
if mode_limited_data == False:
    ds_train_size  = len(fnames_train)    
else:
    # Randomly select slices from the entire training set
    random.shuffle(fnames_train)

# Split test, train and eval 
fnames_test = [fnames_train.pop() for i in range(ds_test_size)]
fnames_eval = [fnames_train.pop() for i in range(ds_eval_size)]

fnames_train = fnames_train[:ds_train_size]

# Transform data
#
# Create a Dataset whose elements are generated by generator
# Uses tf.numpy_function (should not use this method if you need to serialize your model and restore it in a different environment.)
# (output_shapes, output_types) will be removed in a future version -->
# Use output_signature instead!!!
# output_signature=(
#         tf.TensorSpec(shape=(), dtype=tf.int32),
#         tf.RaggedTensorSpec(shape=(2, None), dtype=tf.int32)))
# ds_train = tf.data.Dataset.from_generator(lambda : gen(fnames=fnames_train),
#     output_signature=(tf.TensorSpec(shape=img_shape_load, dtype=tf.float32)))
# ds_eval = tf.data.Dataset.from_generator(lambda : gen(fnames=fnames_eval), 
#     (tf.float32), output_shapes=(tf.TensorShape(img_shape_load)))
# ds_test = tf.data.Dataset.from_generator(lambda : gen(fnames=fnames_test), 
#     (tf.float32), output_shapes=(tf.TensorShape(img_shape_load)))                                      
ds_train = tf.data.Dataset.from_generator(lambda : gen(fnames=fnames_train), (tf.float32), output_shapes=(tf.TensorShape(img_shape_load)))
ds_eval = tf.data.Dataset.from_generator(lambda : gen(fnames=fnames_eval), (tf.float32), output_shapes=(tf.TensorShape(img_shape_load)))
ds_test = tf.data.Dataset.from_generator(lambda : gen(fnames=fnames_test), (tf.float32), output_shapes=(tf.TensorShape(img_shape_load)))

# ds_train = ds_train.apply()

""" 
for a in ds_train.take(3):
    plt.imshow(a[50,:,:]); plt.colorbar(); plt.show();
    print(a.get_shape().as_list())

imgs = list(ds_train.take(2))
for img in imgs:
    plt.imshow(img[50,:,:]); plt.colorbar(); plt.show();
    print(img.get_shape().as_list())  
    
list(dataset.as_numpy_iterator())
    
"""
    
# Add noise
ds_train = ds_train.map(  # return tuple (noisy, target) for training
        lambda x: (add_gaussian_noise(x), x),
        num_parallel_calls = tf.data.experimental.AUTOTUNE)
ds_eval = ds_eval.map(  # return tuple (noisy, target) for training
        lambda x: (add_gaussian_noise(x), x),
        num_parallel_calls = tf.data.experimental.AUTOTUNE)
ds_test = ds_test.map(  # return tuple (noisy, target) for training
        lambda x: (add_gaussian_noise(x), x),
        num_parallel_calls = tf.data.experimental.AUTOTUNE)

print('Examples of training data random crops')
for a,b in ds_train.take(2):
    displaySlices([a.numpy()[0,:,:],b.numpy()[0,:,:] ], 
               ['Noisy image', 'Raw image'], figsize=(12,4))   

# Buffer size greater than or equal to the full size of the dataset 
# buffer_size_train = ds_train.cardinality().numpy()
# buffer_size_eval = ds_eval.cardinality().numpy()

# DS: .cache() keeps the images in memory after they're loaded off disk during 
# the first epoch. If dataset is too large (also use this method to create 
# a performant on-disk cache.
# .prefetch() overlaps data preprocessing and model execution while training
# To randomize iteration order, call shuffle after calling cache
# 
# Apply time consuming operations before cache
ds_train = ds_train.cache()   
ds_train = ds_train.map(random_crop_two_imgs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train_batched = ds_train.batch(batch_size)
ds_train_batched = ds_train_batched.prefetch(tf.data.experimental.AUTOTUNE)  
# 
ds_eval = ds_eval.cache()   
ds_eval = ds_eval.map(random_crop_two_imgs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_eval_batched = ds_eval.batch(batch_size)
ds_eval_batched = ds_eval_batched.prefetch(tf.data.experimental.AUTOTUNE)  

ds_test = ds_test.map(random_crop_two_imgs, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Images eval for tensorboard summary
images_eval = []
images_eval_noisy = []
count = 0
for a_n, a in ds_eval_batched.take(1):
    images_eval = a.numpy()
    images_eval_noisy = a_n.numpy()


"""
# Write images to tensorboard
# Create a file writer 
log_dir = get_run_logdir(path_results)
file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')
images_eval_pred = model.predict(images_eval_noisy)

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

"""
# Test data
data_test, data_test_noisy = get_imgs_from_dataset(ds_test, ds_test_size)

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
    if conv_model == "3DCNN": # UNet, Convnet
        model   = get_model_simple_convnet(
            input_shape=input_shape, 
            conv_filt=conv_filt,
            kernel_size=kernel_size,
            activation=activation,
            padding=padding)
    elif conv_model == "3DUNet":    
        model   = get_model_unet(
            input_shape=input_shape, 
            conv_filt=conv_filt,
            kernel_size=kernel_size,
            activation=activation,
            padding=padding,
            pool_size=pool_size)

    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)    
    model.compile(optimizer=optimizer,
              loss=loss_fn, 
              metrics=["mse", 
                       "mae", # equivalent to mape for [0,1] images, mape not working properly
                       #"mean_absolute_percentage_error" # mape = tf.keras.losses.MeanAbsolutePercentageError()
                       ])         
         
# TRAIN
#
# Fit the model
history = model.fit(ds_train_batched, 
                    epochs=epochs, 
                    validation_data=ds_eval_batched,callbacks=callbacks)


# Losses
fig = pd.DataFrame(history.history).plot(figsize=[8, 5], logy=False).get_figure()
plt.show()
fig.savefig(os.path.join(log_dir,
                            name_save+'_losses.png'))

# -------------------------------------------------------
# PREDICT AND ERROR
# 
# Load best model
model = keras.models.load_model(os.path.join(path_results, 
                                              name_save+'_model_best.h5'))
                  
# Test data
data_test, data_test_noisy = get_imgs_from_dataset(ds_test, batch_size)

"""
# Save test data
np.savez(os.path.join(path_data_main,
                            'Kits19_256x_3D_test_data_5pcnoise_crop_rnd' + '.npz'), 
                          data_test = data_test, data_test_noisy=data_test_noisy)    
"""
# Predict (denoise image)
data_pred = model.predict(data_test_noisy)

data_test_norm = np.sqrt(np.sum(np.square(data_test)))
data_pred_err = np.sqrt(np.sum(np.square(data_test-data_pred)))/data_test_norm

print('Error %.2f' % (100.0*data_pred_err))

# Display result
print('Example of restored images')
for i in range(5):
    displaySlices([data_test[i,0,:,:,0], data_test_noisy[i,0,:,:,0], 
                   data_pred[i,0,:,:,0]], 
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
