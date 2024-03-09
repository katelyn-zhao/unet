from simple_unet_model import simple_unet_model   #Use normal unet model
from keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import random
from keras.callbacks import Callback


image_directory = 'MRI/slices/img/'
mask_directory = 'MRI/slices/mask/'

SIZE = 128
image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

images = os.listdir(image_directory)
for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'png'):
        #print(image_directory+image_name)
        image = cv2.imread(image_directory+image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(mask_directory+image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))

#Normalize images
image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1),3)
#D not normalize masks, just rescale to 0 to 1.
mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.

X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.20, random_state = 0)

# #Sanity check, view few mages
# # image_number = random.randint(0, len(X_train))
# # plt.figure(figsize=(12, 6))
# # plt.subplot(121)
# # plt.imshow(np.reshape(X_train[image_number], (256, 256)), cmap='gray')
# # plt.subplot(122)
# # plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
# # plt.show()

##############################################################################

class DiceScoreCallback(Callback):
    def __init__(self, validation_data):
        super(DiceScoreCallback, self).__init__()
        self.validation_data = validation_data
    
    def dice_coef(self, y_true, y_pred, smooth=1e-5):
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred)
        return (2.0 * intersection + smooth) / (union + smooth)
    
    def on_epoch_end(self, epoch, logs=None):
        all_dice_scores = []
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_pred = self.model.predict(X_val)
        dice_scores = []
        for i in range(len(y_val)):
            dice_score = self.dice_coef(y_val[i], y_pred[i])
            dice_scores.append(dice_score)
        mean_dice_score = np.mean(dice_scores)
        print(f'Epoch {epoch + 1} - Dice Score: {mean_dice_score:.4f}')
        all_dice_scores.append(mean_dice_score)

dice_score_callback = DiceScoreCallback(validation_data=(X_test, y_test))

# ###############################################################
IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]

def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = get_model()

#If starting with pre-trained weights. 
model.load_weights('liver.hdf5')

history = model.fit(X_train, y_train, 
                    batch_size = 16, 
                    verbose=1, 
                    epochs=20, 
                    validation_data=(X_test, y_test), 
                    shuffle=False,
                    callbacks = [dice_score_callback])

model.save('liver.hdf5')

# ############################################################

#DICE
y_pred = model.predict(X_test)
y_pred_thresholded = y_pred > 0.5

smooth = 1e-5
intersection = (np.logical_and(y_test, y_pred_thresholded))
union = y_pred.sum() + y_test.sum()
dice_score = ((2. * np.sum(intersection)) + smooth) / (union + smooth)
print("Average Dice score is: ", dice_score)

#######################################################################
#Predict on a few images
model = get_model()
model.load_weights('liver.hdf5') #60 epochs

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

# test_img_other = cv2.imread('data/test_images/02-1_256.tif', 0)
# #test_img_other = cv2.imread('data/test_images/img8.tif', 0)
# test_img_other_norm = np.expand_dims(normalize(np.array(test_img_other), axis=1),2)
# test_img_other_norm=test_img_other_norm[:,:,0][:,:,None]
# test_img_other_input=np.expand_dims(test_img_other_norm, 0)

# #Predict and threshold for values above 0.5 probability
# #Change the probability threshold to low value (e.g. 0.05) for watershed demo.
# prediction_other = (model.predict(test_img_other_input)[0,:,:,0] > 0.2).astype(np.uint8)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')
plt.subplot(234)
plt.title('Testing Image')
plt.imshow(test_img2[:,:,0], cmap='gray')
plt.subplot(235)
plt.title('Testing Label')
plt.imshow(ground_truth2[:,:,0], cmap='gray')
plt.subplot(236)
plt.title("Prediction on test image")
plt.imshow(prediction2, cmap='gray')
plt.show()