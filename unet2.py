import os
import io
import random
import nibabel
import numpy as np
from glob import glob
import nibabel as nib
import tensorflow as tf
from nibabel import load
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


image_directory = 'MRI/Anatomical_mag_echo5/'
mask_directory = 'MRI/whole_liver_segmentation/'

data_output_path = 'MRI/slices/'
image_slice_output = os.path.join(data_output_path, 'img/')
mask_slice_output = os.path.join(data_output_path, 'mask/')

image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

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
sliced_image_dataset = np.expand_dims(normalize(np.array(sliced_image_dataset), axis=1),3)
#D not normalize masks, just rescale to 0 to 1.
sliced_mask_dataset = np.expand_dims((np.array(sliced_mask_dataset)),3) /255.


X_train, X_test, y_train, y_test = train_test_split(sliced_image_dataset, sliced_mask_dataset, test_size = 0.20, random_state = 0)


#Sanity check, view few mages
# image_number = random.randint(0, len(X_train))
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(X_train[image_number], cmap='gray')
# plt.subplot(122)
# plt.imshow(y_train[image_number], cmap='gray')
# plt.show()

##############################################################################

class DiceScoreCallback(Callback):
    def __init__(self, validation_data, save_path):
        super(DiceScoreCallback, self).__init__()
        self.validation_data = validation_data
        self.save_path = save_path
        self.dice_scores = []
    
    def dice_coef(self, y_true, y_pred, smooth=1.):
        intersection = (np.logical_and(y_true, y_pred))
        union = y_true.sum() + y_pred.sum()
        return ((2. * np.sum(intersection)) + smooth) / (union + smooth)
    
    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_pred = model.predict(X_val)
        y_pred = y_pred > 0.5
        dice_scores = []
        for i in range(len(y_val)):
            dice_score = self.dice_coef(y_val[i], y_pred[i])
            dice_scores.append(dice_score)
        mean_dice_score = np.mean(dice_scores)
        dice_scores.append(mean_dice_score)
        print(f'Epoch {epoch + 1} - Dice Score: {mean_dice_score:.4f}')
        
        # Save the dice scores to a file
        with open(self.save_path, 'a') as f:
            f.write(str(mean_dice_score) + "\n")

save_path = 'dice_scores.txt'
dice_score_callback = DiceScoreCallback(validation_data=(X_test, y_test), save_path=save_path)

##############################################################################

IMG_HEIGHT = sliced_image_dataset.shape[1]
IMG_WIDTH  = sliced_image_dataset.shape[2]
IMG_CHANNELS = sliced_image_dataset.shape[3]

def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = get_model()

#If starting with pre-trained weights. 
#model.load_weights('liver.hdf5')

history = model.fit(X_train, y_train, 
                    batch_size = 16, 
                    verbose=1, 
                    epochs=20, 
                    validation_data=(X_test, y_test), 
                    shuffle=False,
                    callbacks=[dice_score_callback])

model.save('liver.hdf5')

##############################################################

# def load_dice_scores(file_path):
#     dice_scores = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             dice_scores.append(float(line))
#     return dice_scores

# save_path = 'dice_scores.txt'
# dice_scores = load_dice_scores(save_path)
# epochs = range(1, len(dice_scores) + 1)
# plt.plot(epochs, dice_scores, 'b', label='Dice Score')
# plt.title('Dice Score vs. Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Dice Score')
# plt.legend()
# plt.grid(True)
# plt.show()

##############################################################

#Predict on a few images
# model = get_model()
# model.load_weights('liver.hdf5') #0 epochs

# test_img_number = random.randint(0, len(X_test))
# test_img = X_test[test_img_number]
# ground_truth=y_test[test_img_number]
# test_img_norm=test_img[:,:,0][:,:,None]
# test_img_input=np.expand_dims(test_img_norm, 0)
# prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

# test_img_number2 = random.randint(0, len(X_test))
# test_img2 = X_test[test_img_number2]
# ground_truth2 = y_test[test_img_number2]
# test_img_norm2 =test_img2[:,:,0][:,:,None]
# test_img_input2 = np.expand_dims(test_img_norm2, 0)
# prediction2 = (model.predict(test_img_input2)[0,:,:,0] > 0.5).astype(np.uint8)

# plt.figure(figsize=(16, 8))
# plt.subplot(231)
# plt.title('Testing Image')
# plt.imshow(test_img[:,:,0], cmap='gray')
# plt.subplot(232)
# plt.title('Testing Label')
# plt.imshow(ground_truth[:,:,0], cmap='gray')
# plt.subplot(233)
# plt.title('Prediction on test image')
# plt.imshow(prediction, cmap='gray')
# plt.subplot(234)
# plt.title('Testing Image')
# plt.imshow(test_img2[:,:,0], cmap='gray')
# plt.subplot(235)
# plt.title('Testing Label')
# plt.imshow(ground_truth2[:,:,0], cmap='gray')
# plt.subplot(236)
# plt.title("Prediction on test image")
# plt.imshow(prediction2, cmap='gray')
# plt.show()