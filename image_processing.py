import os
import io
import random
import nibabel
import cv2
from scipy.ndimage import rotate
import numpy as np
import nibabel as nib
from nibabel import load
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from keras.utils import Sequence
from IPython.display import Image, display
from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import normalize
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from sklearn.model_selection import KFold
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda


image_directory = 'MRI/Anatomical_mag_echo5/'
mask_directory = 'MRI/whole_liver_segmentation/'

image_dataset = []  
mask_dataset = []
sliced_image_dataset = []
sliced_mask_dataset = []

# SIZE = 128

images = os.listdir(image_directory)
for i, image_name in enumerate(images):    
    if (image_name.split('.')[1] == 'nii'):
        image = nib.load(image_directory+image_name)
        image = np.array(image.get_fdata())
        #image = resize(image, (SIZE, SIZE))
        image_dataset.append(np.array(image))
        
masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'nii'):
        image = nib.load(mask_directory+image_name)
        image = np.array(image.get_fdata())
        #image = resize(image, (SIZE, SIZE))
        mask_dataset.append(np.array(image))

for i in range(len(image_dataset)):
    for j in range(image_dataset[i].shape[2]):
        sliced_image_dataset.append(image_dataset[i][:,:,j])
        sliced_mask_dataset.append(mask_dataset[i][:,:,j])
        #Adding randomly rotated slices
        cw = random.randint(0,1)
        angle = random.randint(5, 10)
        if cw:
            sliced_image_dataset.append(rotate(image_dataset[i][:,:,j], angle, reshape = False))
            sliced_mask_dataset.append(rotate(mask_dataset[i][:,:,j], angle, reshape = False))
        else:
            sliced_image_dataset.append(rotate(image_dataset[i][:,:,j], angle * -1, reshape = False))
            sliced_mask_dataset.append(rotate(mask_dataset[i][:,:,j], angle * -1, reshape = False))
        #contrast adjustment
        adjust = random.randint(0,1)
        contrast = random.randint(1, 2)
        if adjust:
            sliced_image_dataset.append(cv2.convertScaleAbs(image_dataset[i][:,:,j], alpha = contrast, beta = 0))
            sliced_mask_dataset.append(cv2.convertScaleAbs(mask_dataset[i][:,:,j], alpha = contrast, beta = 0))
        if adjust and cw:
            sliced_image_dataset.append(rotate(cv2.convertScaleAbs(image_dataset[i][:,:,j], alpha = contrast, beta = 0), angle, reshape = False))
            sliced_mask_dataset.append(rotate(cv2.convertScaleAbs(mask_dataset[i][:,:,j], alpha = contrast, beta = 0), angle, reshape = False))

#Normalize images
sliced_image_dataset = np.expand_dims(np.array(sliced_image_dataset),3)
#D not normalize masks, just rescale to 0 to 1.
sliced_mask_dataset = np.expand_dims((np.array(sliced_mask_dataset)),3)

print(len(sliced_image_dataset))

X_train, X_test, y_train, y_test = train_test_split(sliced_image_dataset, sliced_mask_dataset, test_size = 0.20, random_state = 0)

#Sanity check, view a few images
image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(X_train[image_number], cmap='gray')
plt.subplot(122)
plt.imshow(y_train[image_number], cmap='gray')
plt.show()