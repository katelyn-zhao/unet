import os
import io
import random
import nibabel
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import rotate
from nibabel import load
from IPython.display import Image, display
from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from sklearn.model_selection import KFold
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda


def dice_coef(y_true, y_pred, smooth=1.):
  y_true_f = tf.keras.backend.flatten(y_true)
  y_pred_f = tf.keras.backend.flatten(y_pred)
  intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
  return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_coef_p(y_true, y_pred, smooth=1.):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return (2.0 * intersection + smooth) / (union + smooth)

def tpr_p(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold)
   
    tp = np.sum((y_pred == 1) & (y_true == 1))

    fn = np.sum((y_pred == 0) & (y_true == 1))

    if (tp == 0):
        tpr = 0
    else:
        tpr = tp / (tp + fn)

    return tpr


def fpr_p(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold)

    fp = np.sum((y_pred == 1) & (y_true == 0))

    tn = np.sum((y_pred == 0) & (y_true == 0))
    
    if (fp == 0):
        fpr = 0
    else:
        fpr = fp / (fp + tn)

    return fpr

def tpr(y_true, y_pred, threshold=0.5):
    y_pred_pos = tf.cast(y_pred > threshold, tf.float32)
    y_true_pos = tf.cast(y_true > threshold, tf.float32)
    
    true_pos = tf.reduce_sum(tf.cast(tf.logical_and(y_true_pos == 1, y_pred_pos == 1), tf.float32))
    actual_pos = tf.reduce_sum(tf.cast(y_true_pos, tf.float32))
    
    tpr = true_pos / (actual_pos + tf.keras.backend.epsilon())
    return tpr

def fpr(y_true, y_pred, threshold=0.5):
    y_pred_pos = tf.cast(y_pred > threshold, tf.float32)
    y_true_neg = tf.cast(y_true <= threshold, tf.float32)
    
    false_pos = tf.reduce_sum(tf.cast(tf.logical_and(y_true_neg == 1, y_pred_pos == 1), tf.float32))
    actual_neg = tf.reduce_sum(tf.cast(y_true_neg, tf.float32))
    
    fpr = false_pos / (actual_neg + tf.keras.backend.epsilon())
    return fpr



################################################################################################################################

def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    optimizer = Adam(learning_rate=1e-3)
    model.compile(optimizer='adam', loss='bce', metrics=[dice_coef, tpr, fpr])
    
    model.summary()
    
    return model

################################################################################################################################

image_directory = 'MRI19/Anatomical_mag_echo5/'
mask_directory = 'MRI19/whole_liver_segmentation/'
peds_image_directory = 'MRI19/PEDIATRIC/Anatomical_mag_echo5/'
peds_mask_directory = 'MRI19/PEDIATRIC/whole_liver_segmentations/'

image_dataset = []  
mask_dataset = []
sliced_image_dataset = []
sliced_mask_dataset = []
image_names = []
sliced_image_names = []

def pad_volume(volume, desired_size=(256, 256, 40)):
    # Calculate the padding needed to reach the desired size
    pad_width = [(max(0, (s - volume.shape[i]) // 2), max(0, (s - volume.shape[i] + 1) // 2)) for i, s in enumerate(desired_size)]
    # Pad the volume
    padded_volume = np.pad(volume, pad_width, mode='constant', constant_values=0)
    return padded_volume

images = os.listdir(image_directory)
for i, image_name in enumerate(images):    
    if (image_name.split('.')[1] == 'nii'):
        image = nib.load(image_directory+image_name)
        image = np.array(image.get_fdata())
        image = pad_volume(image, (256, 256, 40))
        image_dataset.append(np.array(image))
        image_names.append(image_name.split('.')[0])

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'nii'):
        image = nib.load(mask_directory+image_name)
        image = np.array(image.get_fdata())
        image = pad_volume(image, (256, 256, 40))
        mask_dataset.append(np.array(image))

# peds_images = os.listdir(peds_image_directory)
# for i, image_name in enumerate(peds_images):    
#     if (image_name.split('.')[1] == 'nii'):
#         image = nib.load(peds_image_directory+image_name)
#         image = np.array(image.get_fdata())
#         image_dataset.append(np.array(image))
#         image_names.append(image_name.split('.')[0])

# peds_masks = os.listdir(peds_mask_directory)
# for i, image_name in enumerate(peds_masks):
#     if (image_name.split('.')[1] == 'nii'):
#         image = nib.load(peds_mask_directory+image_name)
#         image = np.array(image.get_fdata())
#         mask_dataset.append(np.array(image))

for i in range(len(image_dataset)):
    for j in range(image_dataset[i].shape[2]):
        sliced_image_dataset.append(image_dataset[i][:,:,j])
        sliced_mask_dataset.append(mask_dataset[i][:,:,j])
        sliced_image_names.append(image_names[i] + '-' + str(j))
        #rotation
        cw = random.randint(0,1)
        angle = random.randint(5,10)
        #contrast adjustment
        adjust = random.randint(0,1)
        contrast = random.randint(1,2)
        #applying changes
        if adjust and cw == 1:
            sliced_image_dataset.append(rotate(cv2.convertScaleAbs(image_dataset[i][:,:,j], alpha = contrast, beta = 0), angle, reshape = False))
            sliced_mask_dataset.append(rotate(mask_dataset[i][:,:,j], angle, reshape = False) > 0.5)
            sliced_image_names.append(image_names[i] + '-' + str(j) + '-aug')
        if adjust and cw == 0:
            sliced_image_dataset.append(rotate(cv2.convertScaleAbs(image_dataset[i][:,:,j], alpha = contrast, beta = 0), angle * -1, reshape = False))
            sliced_mask_dataset.append(rotate(mask_dataset[i][:,:,j], angle * -1, reshape = False) > 0.5)
            sliced_image_names.append(image_names[i] + '-' + str(j) + '-aug')

#Normalize images
sliced_image_dataset = np.expand_dims(np.array(sliced_image_dataset),3)
#D not normalize masks, just rescale to 0 to 1.
sliced_mask_dataset = np.expand_dims((np.array(sliced_mask_dataset)),3)

f = open(f"C:/Users/Mittal/Desktop/kunet/optimized7_output.txt", "a")
print("sliced image dataset: ", len(sliced_image_dataset), file=f)
f.close()

#Sanity check, view a few images
# image_number = random.randint(0, len(X_train))
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(X_train[image_number], cmap='gray')
# plt.subplot(122)
# plt.imshow(y_train[image_number], cmap='gray')
# plt.show()

##############################################################################################################################################

IMG_HEIGHT = sliced_image_dataset.shape[1]
IMG_WIDTH  = sliced_image_dataset.shape[2]
IMG_CHANNELS = sliced_image_dataset.shape[3]

def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

n_splits = 5

kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

# Iterate over each fold
for i, (train_index, test_index) in enumerate(kf.split(sliced_image_dataset, sliced_mask_dataset)):
    if i == 3:
        break
    else:
        X_train, X_test = sliced_image_dataset[train_index], sliced_image_dataset[test_index]
        y_train, y_test = sliced_mask_dataset[train_index], sliced_mask_dataset[test_index]
        name_test = np.array(sliced_image_names)[test_index]

        f = open(f"C:/Users/Mittal/Desktop/kunet/optimized7_output.txt", "a")
        print("FOLD----------------------------------", file=f)
        print("x-training: ", len(X_train), file=f)
        print("x-testing: ", len(X_test), file=f)
        print("y-training: ", len(y_train), file=f)
        print("y-testing: ", len(y_test), file=f)
        f.close()

        model = get_model()

        checkpoint = ModelCheckpoint(f'C:/Users/Mittal/Desktop/kunet/optimized7_best_model{i}.keras', monitor='val_loss', save_best_only=True)

        history = model.fit(X_train, y_train,
                            batch_size=16,
                            verbose=1,
                            epochs=800,
                            validation_data=(X_test, y_test),
                            shuffle=False,
                            callbacks = [checkpoint])

        plt.figure(figsize=(15,5))
        plt.subplot(1,2,1)
        plt.plot(history.history['loss'], color='r')
        plt.plot(history.history['val_loss'])
        plt.ylabel('Losses')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val.'], loc='upper right')
        plt.subplot(1,2,2)
        plt.plot(history.history['dice_coef'], color='r')
        plt.plot(history.history['val_dice_coef'])
        plt.ylabel('dice_coef')
        plt.xlabel('Epoch')
        plt.tight_layout()
        plt.savefig(f'C:/Users/Mittal/Desktop/kunet/optimized7_process{i}.png')
        plt.close()

        max_dice_coef = max(history.history['dice_coef'])
        max_val_dice_coef = max(history.history['val_dice_coef'])
        max_tpr = max(history.history['tpr'])
        min_fpr = min(history.history['fpr'])

        f = open(f"C:/Users/Mittal/Desktop/kunet/optimized7_output.txt", "a")
        print("max dice coef: ", max_dice_coef, file=f)
        print("max val dice coef: ", max_val_dice_coef, file=f)
        print("max tpr: ", max_tpr, file=f)
        print("min fpr: ", min_fpr, file=f)
        f.close()
        
        model.load_weights(f'C:/Users/Mittal/Desktop/kunet/optimized7_best_model{i}.keras')

        dice_scores = []
        tprs = []
        fprs = []

        for z in range(50):
            test_img_number = random.randint(0, len(X_test)-1)
            test_img = X_test[z]
            ground_truth = y_test[z]
            test_img_norm = test_img[:,:,0][:,:,None]
            test_img_input = np.expand_dims(test_img_norm, 0)
            prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.double)

            original_image_normalized = ground_truth.astype(float) / np.max(ground_truth)
            colored_mask = plt.get_cmap('jet')(prediction / np.max(prediction))
            alpha = 0.5 
            colored_mask[..., 3] = np.where(prediction > 0, alpha, 0)

            dice_scores.append(dice_coef(ground_truth, prediction))
            tprs.append(tpr(ground_truth, prediction))
            fprs.append(fpr(ground_truth, prediction))

            plt.figure(figsize=(16, 8))
            plt.subplot(141)
            plt.title('Testing Image')
            plt.imshow(test_img[:,:,0], cmap='gray')
            plt.subplot(142)
            plt.title('Testing Label')
            plt.imshow(ground_truth[:,:,0], cmap='gray')
            plt.subplot(143)
            plt.title('Prediction on test image')
            plt.imshow(prediction, cmap='gray')
            plt.subplot(144)
            plt.title("Overlayed Images")
            plt.imshow(original_image_normalized, cmap='gray')
            plt.imshow(colored_mask, cmap='jet')
            plt.savefig(f'C:/Users/Mittal/Desktop/kunet/optimizedpredict7/fold{i}_{name_test[test_img_number]}.png')
            plt.close()
        
        average_dice_coef = np.mean(dice_scores)
        average_tpr = np.mean(tprs)
        average_fpr = np.mean(fprs)

        f = open(f'C:/Users/Mittal/Desktop/kunet/optimized7_output.txt', "a")
        print('average prediction dice score: ', average_dice_coef, file=f)
        print('average prediction tpr: ', average_tpr, file=f)
        print('average prediction fpr: ', average_fpr, file=f)
        f.close()
