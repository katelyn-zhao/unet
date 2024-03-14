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
image_directory = 'MRI/Anatomical_mag_echo5/'
mask_directory = 'MRI/whole_liver_segmentation/'

image_dataset = []  
mask_dataset = []
sliced_image_dataset = []
sliced_mask_dataset = []

SIZE = 128

images = os.listdir(image_directory)
for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'nii'):
        image = nib.load(image_directory+image_name)
        image = np.array(image.get_fdata())
        image = resize(image, (SIZE, SIZE))
        image_dataset.append(np.array(image))

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'nii'):
        image = nib.load(mask_directory+image_name)
        image = np.array(image.get_fdata())
        image = resize(image, (SIZE, SIZE))
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

##############################################################################

IMG_HEIGHT = sliced_image_dataset.shape[1]
IMG_WIDTH  = sliced_image_dataset.shape[2]
IMG_CHANNELS = sliced_image_dataset.shape[3]

def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = get_model()


model.load_weights('best_model.h5')

test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

test_img_number2 = random.randint(0, len(X_test))
test_img2 = X_test[test_img_number2]
ground_truth2 = y_test[test_img_number2]
test_img_norm2 =test_img2[:,:,0][:,:,None]
test_img_input2 = np.expand_dims(test_img_norm2, 0)
prediction2 = (model.predict(test_img_input2)[0,:,:,0] > 0.5).astype(np.uint8)


original_image_normalized = ground_truth.astype(float) / np.max(ground_truth)
colored_mask = plt.get_cmap('jet')(prediction / np.max(prediction))
alpha = 0.5  # Transparency level
colored_mask[..., 3] = np.where(prediction > 0, alpha, 0)

original_image_normalized2 = ground_truth2.astype(float) / np.max(ground_truth2)
colored_mask2 = plt.get_cmap('jet')(prediction2 / np.max(prediction2))
alpha = 0.5  # Transparency level
colored_mask2[..., 3] = np.where(prediction2 > 0, alpha, 0)



plt.figure(figsize=(16, 8))
plt.subplot(241)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(242)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(243)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')
plt.subplot(244)
plt.title("Overlayed Images")
plt.imshow(original_image_normalized, cmap='gray')
plt.imshow(colored_mask, cmap='jet')
plt.subplot(245)
plt.title('Testing Image')
plt.imshow(test_img2[:,:,0], cmap='gray')
plt.subplot(246)
plt.title('Testing Label')
plt.imshow(ground_truth2[:,:,0], cmap='gray')
plt.subplot(247)
plt.title("Prediction on test image")
plt.imshow(prediction2, cmap='gray')
plt.subplot(248)
plt.title("Overlayed Images")
plt.imshow(original_image_normalized2, cmap='gray')
plt.imshow(colored_mask2, cmap='jet')
plt.show()