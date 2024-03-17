import os
import io
import random
import nibabel
import numpy as np
import nibabel as nib
from nibabel import load
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import Sequence
from IPython.display import Image, display
from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import normalize
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from simple_unet_model import simple_unet_model
from skimage.transform import resize

#if u forget again :)
#https://github.com/mrdbourke/m1-machine-learning-test

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

for i in range(len(mask_dataset)):
    for j in range(mask_dataset[i].shape[2]):
        if i == 16 and j == 25:
            continue
        else:
            sliced_mask_dataset.append(mask_dataset[i][:,:,j])

#Normalize images
sliced_image_dataset = np.expand_dims(np.array(sliced_image_dataset),3)
#D not normalize masks, just rescale to 0 to 1.
sliced_mask_dataset = np.expand_dims((np.array(sliced_mask_dataset)),3)

X_train, X_test, y_train, y_test = train_test_split(sliced_image_dataset, sliced_mask_dataset, test_size = 0.20, random_state = 0)

#Sanity check, view a few images
# image_number = random.randint(0, len(X_train))
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(X_train[image_number], cmap='gray')
# plt.subplot(122)
# plt.imshow(y_train[image_number], cmap='gray')
# plt.show()

##############################################################################

IMG_HEIGHT = sliced_image_dataset.shape[1]
IMG_WIDTH  = sliced_image_dataset.shape[2]
IMG_CHANNELS = sliced_image_dataset.shape[3]

def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = get_model()

checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# If starting with pre-trained weights. 
#model.load_weights('liver.hdf5')

history = model.fit(X_train, y_train, 
                    batch_size = 16, 
                    verbose=1, 
                    epochs=300, 
                    validation_data=(X_test, y_test), 
                    shuffle=False,
                    callbacks = [checkpoint])

model.save('liver.hdf5')