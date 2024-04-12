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

#Predict on a few images
image_directory = 'MRI/prediction/'
mask_directory = 'MRI/prediction labels/'

image_dataset = []  
mask_dataset = []
sliced_image_dataset = []
sliced_mask_dataset = []
image_names = []
sliced_image_names = []

images = os.listdir(image_directory)
for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'nii'):
        image = nib.load(image_directory+image_name)
        image = np.array(image.get_fdata())
        image_dataset.append(np.array(image))
        image_names.append(image_name.split('.')[0])

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'nii'):
        image = nib.load(mask_directory+image_name)
        image = np.array(image.get_fdata())
        mask_dataset.append(np.array(image))

for i in range(len(image_dataset)):
    for j in range(image_dataset[i].shape[2]):
        sliced_image_dataset.append(image_dataset[i][:,:,j])
        sliced_mask_dataset.append(mask_dataset[i][:,:,j])
        sliced_image_names.append(image_names[i] + '-' + str(j))

#Normalize images
sliced_image_dataset = np.expand_dims(np.array(sliced_image_dataset),3)
#D not normalize masks, just rescale to 0 to 1.
sliced_mask_dataset = np.expand_dims((np.array(sliced_mask_dataset)),3)

##############################################################################

IMG_HEIGHT = sliced_image_dataset.shape[1]
IMG_WIDTH  = sliced_image_dataset.shape[2]
IMG_CHANNELS = sliced_image_dataset.shape[3]

def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = get_model()

model.load_weights('optimized6_best_model2.keras')

dice_scores = []

for i in range(len(sliced_image_dataset)):
    test_img = sliced_image_dataset[i]
    ground_truth= sliced_mask_dataset[i]
    test_img_norm=test_img[:,:,0][:,:,None]
    test_img_input=np.expand_dims(test_img_norm, 0)
    prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

    original_image_normalized = ground_truth.astype(float) / np.max(ground_truth)
    colored_mask = plt.get_cmap('jet')(prediction / np.max(prediction))
    alpha = 0.5  # Transparency level
    colored_mask[..., 3] = np.where(prediction > 0, alpha, 0)

    plt.figure(figsize=(8, 8))
    # plt.subplot(141)
    # plt.title('Testing Image')
    # plt.imshow(test_img[:,:,0], cmap='gray')
    # plt.subplot(142)
    # plt.title('Testing Label')
    # plt.imshow(ground_truth[:,:,0], cmap='gray')
    # plt.subplot(143)
    plt.imshow(prediction, cmap='gray')
    plt.axis('off')
    # plt.subplot(144)
    # plt.title("Overlayed Images")
    # plt.imshow(original_image_normalized, cmap='gray')
    # plt.imshow(colored_mask, cmap='jet')
    plt.savefig(f'kunet/prediction slice only/{sliced_image_names[i]}.png')
    plt.close()