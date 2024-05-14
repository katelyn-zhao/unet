import random
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import nibabel
import cv2
import numpy as np
import nibabel as nib
from nibabel import load
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import rotate
from IPython.display import Image, display
from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.utils import normalize
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from sklearn.model_selection import KFold
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def dice_coef(y_true, y_pred):
    num_classes = 10
    total_dice = 0.0
    num_class = 0.0
    for class_idx in range(num_classes):
        y_true_class = y_true[..., class_idx]
        y_pred_class = y_pred[..., class_idx]
        y_true_f = tf.keras.backend.flatten(y_true_class)
        y_pred_f = tf.keras.backend.flatten(y_pred_class)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        dice = (2. * intersection) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1e-7)
        total_dice = total_dice + dice
        num_class = num_class + 1.0
    mean_dice_score = total_dice / num_class
    return mean_dice_score

def tpr(y_true, y_pred, threshold=0.5):
    num_classes = 10
    total_tpr = 0
    num_class = 0
    for class_idx in range(num_classes):
        y_true_class = y_true[..., class_idx]
        y_pred_class = y_pred[..., class_idx]
        y_pred_pos = tf.cast(y_pred_class > threshold, tf.float32)
        y_true_pos = tf.cast(y_true_class > threshold, tf.float32)
        true_pos = tf.reduce_sum(tf.cast(tf.logical_and(y_true_pos == 1, y_pred_pos == 1), tf.float32))
        actual_pos = tf.reduce_sum(tf.cast(y_true_pos, tf.float32))
        tpr = true_pos / (actual_pos + tf.keras.backend.epsilon())
        total_tpr += tpr
        num_class += 1
    mean_tpr = total_tpr / num_class
    return mean_tpr

def fpr(y_true, y_pred, threshold=0.5):
    num_classes = 10
    total_fpr = 0
    num_class = 0
    for class_idx in range(num_classes):
        y_true_class = y_true[..., class_idx]
        y_pred_class = y_pred[..., class_idx]
        y_pred_pos = tf.cast(y_pred_class > threshold, tf.float32)
        y_true_neg = tf.cast(y_true_class <= threshold, tf.float32)
        false_pos = tf.reduce_sum(tf.cast(tf.logical_and(y_true_neg == 1, y_pred_pos == 1), tf.float32))
        actual_neg = tf.reduce_sum(tf.cast(y_true_neg, tf.float32))
        fpr = false_pos / (actual_neg + tf.keras.backend.epsilon())
        total_fpr += fpr
        num_class += 1
    mean_fpr = total_fpr / num_class
    return mean_fpr



##############################################################################################

def multi_unet_model(n_classes=9, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
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
     
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[dice_coef, tpr, fpr])
    model.summary()
    
    return model

##############################################################################################

#Number of classes for segmentation
n_classes= 10

#Capture training image info as a list
image_directory = 'MRI19/multi_Anatomical_mag_echo5/'
mask_directory = 'MRI19/multi_liver_segmentation/'

image_dataset = []  
mask_dataset = []
sliced_image_dataset = []
sliced_mask_dataset = []
image_names = []
sliced_image_names = []


images = sorted(os.listdir(image_directory))
for i, image_name in enumerate(images):    
    if (image_name.split('.')[1] == 'nii'):
        image = nib.load(image_directory+image_name)
        image = np.array(image.get_fdata())
        image_dataset.append(np.array(image))
        image_names.append(image_name.split('.')[0])

masks = sorted(os.listdir(mask_directory))
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
    

sliced_image_dataset = np.array(sliced_image_dataset)
sliced_mask_dataset = np.array(sliced_mask_dataset)
image_names = np.array(image_names)
sliced_image_names = np.array(sliced_image_names)

#Sanity check, view a few images
# image_number = random.randint(0, len(sliced_image_dataset))
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(sliced_image_dataset[image_number], cmap='gray')
# plt.subplot(122)
# plt.imshow(sliced_mask_dataset[image_number], cmap='gray')
# plt.show()

#Encode labels... but multi dim array so need to flatten, encode and reshape
labelencoder = LabelEncoder()
n, h, w = sliced_mask_dataset.shape
sliced_masks_reshaped = sliced_mask_dataset.reshape(-1,1)
sliced_masks_reshaped_encoded = labelencoder.fit_transform(sliced_masks_reshaped)
sliced_masks_encoded_original_shape = sliced_masks_reshaped_encoded.reshape(n, h, w)

print(np.unique(sliced_masks_encoded_original_shape))

sliced_image_dataset = np.expand_dims(sliced_image_dataset, axis=3)
sliced_image_dataset = normalize(sliced_image_dataset, axis=1)

sliced_mask_dataset = np.expand_dims(sliced_masks_encoded_original_shape, axis=3)

##############################################################################################

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

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
        y_train_cat = to_categorical(y_train, num_classes=n_classes)
        y_test_cat = to_categorical(y_test, num_classes=n_classes)

        IMG_HEIGHT = X_train.shape[1]
        IMG_WIDTH  = X_train.shape[2]
        IMG_CHANNELS = X_train.shape[3]

        model = get_model()

        checkpoint = ModelCheckpoint(f'C:/Users/Mittal/Desktop/kunet/multikunet/multi_kunet{i}.h5', monitor='val_loss', save_best_only=True)

        history = model.fit(X_train, y_train_cat, 
                            batch_size=16, 
                            verbose=1, 
                            epochs=300, 
                            validation_data=(X_test, y_test_cat), 
                            shuffle=False,
                            callbacks=[checkpoint])
                            
        #Evaluate the model
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
        plt.savefig(f'C:/Users/Mittal/Desktop/kunet/multikunet/multikunet_process{i}.png')
        plt.close()

        max_dice_coef = max(history.history['dice_coef'])
        max_val_dice_coef = max(history.history['val_dice_coef'])
        max_tpr = max(history.history['tpr'])
        max_val_tpr = max(history.history['val_tpr'])
        min_tpr = max(history.history['fpr'])
        min__val_tpr = min(history.history['val_fpr'])

        f = open(f'C:/Users/Mittal/Desktop/kunet/multikunet/output.txt', "a")
        print("FOLD------------------------------------------", file=f)
        print("Max Dice Score: ", max_dice_coef, file=f)
        print("Max Val Dice Score: ", max_val_dice_coef, file=f)
        print("Max TPR: ", max_tpr, file=f)
        print("Max Val TPR: ". max_val_tpr, file=f)
        print("Max FPR: ", max_fpr, file=f)
        print("Max Val FPR: ". max_val_fpr, file=f)
        f.close()
            
        model.load_weights(f'C:/Users/Mittal/Desktop/kunet/multikunet/multi_kunet{i}.h5')

        dice_scores = []

        for z in range(25):
            test_img_number = random.randint(0, len(X_test)-1)
            test_img = X_test[test_img_number]
            ground_truth = y_test[test_img_number]
            test_img_norm = test_img[:,:,0][:,:,None]
            test_img_input = np.expand_dims(test_img_norm, 0)
            prediction = (model.predict(test_img_input))
            predicted_img = np.argmax(prediction, axis=3)[0,:,:]

            # shape = prediction.shape
            # prediction_dice = prediction

            # for row in range(shape[0]): 
            #     for col in range(shape[1]):
            #         if (prediction[0, row, col, 0] <= 0.5):
            #             prediction_dice[row, col] == 0

            # prediction_dice = prediction_dice.astype(np.float32)
            # ground_truth_dice = ground_truth.astype(np.float32)

            # dice_score = dice_coef(ground_truth_dice, prediction_dice)
            # dice_scores.append(dice_score)

            # original_image_normalized = ground_truth.astype(float) / np.max(ground_truth)
            # colored_mask = plt.get_cmap('jet')(prediction / np.max(prediction))
            # alpha = 0.5 
            # colored_mask[..., 3] = np.where(prediction > 0, alpha, 0)

            plt.figure(figsize=(16, 8))
            plt.subplot(131)
            plt.title('Testing Image')
            plt.imshow(test_img[:,:,0], cmap='gray')
            plt.subplot(132)
            plt.title('Testing Label')
            plt.imshow(ground_truth[:,:,0], cmap='jet')
            plt.subplot(133)
            plt.title('Prediction on test image')
            plt.imshow(predicted_img, cmap='jet')
            # plt.subplot(144)
            # plt.title("Overlayed Images")
            # plt.imshow(original_image_normalized, cmap='jet')
            # plt.imshow(colored_mask, cmap='jet')
            plt.savefig(f'C:/Users/Mittal/Desktop/kunet/multikunet/predict/fold{i}_{z}.png')
            plt.close()

        f = open(f'C:/Users/Mittal/Desktop/kunet/multikunet/output.txt', "a")
        print("Average Prediction Dice Score: ", np.mean(dice_scores), file=f)
        f.close()

##############################################################################################
# ##############################################################################################

# from sklearn.utils import class_weight
# class_weights = class_weight.compute_class_weight('balanced', np.unique(sliced_masks_reshaped_encoded), sliced_masks_reshaped_encoded)

# f = open(f'C:/Users/Mittal/Desktop/kunet/multikunet/output_with_weights.txt', "a")
# print("Class weights are...:", class_weights, file=f)
# f.close()

# IMG_HEIGHT = X_train.shape[1]
# IMG_WIDTH  = X_train.shape[2]
# IMG_CHANNELS = X_train.shape[3]

# def get_model():
#     return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

# model = get_model()

# checkpoint = ModelCheckpoint(f'C:/Users/Mittal/Desktop/kunet/multikunet/multi_kunet0_with_weights.h5', monitor='val_loss', save_best_only=True)

# history = model.fit(X_train, y_train_cat, 
#                     batch_size=16, 
#                     verbose=1, 
#                     epochs=5, 
#                     validation_data=(X_test, y_test_cat), 
#                     class_weight=class_weights,
#                     shuffle=False,
#                     callbacks=[checkpoint])
                    
# #Evaluate the model
# plt.figure(figsize=(15,5))
# plt.subplot(1,2,1)
# plt.plot(history.history['loss'], color='r')
# plt.plot(history.history['val_loss'])
# plt.ylabel('Losses')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Val.'], loc='upper right')
# plt.subplot(1,2,2)
# plt.plot(history.history['dice_coef'], color='r')
# plt.plot(history.history['val_dice_coef'])
# plt.ylabel('dice_coef')
# plt.xlabel('Epoch')
# plt.tight_layout()
# plt.savefig(f'C:/Users/Mittal/Desktop/kunet/multikunet/multikunet_process0_with_weights.png')
# plt.close()

# max_dice_coef = max(history.history['dice_coef'])
# max_val_dice_coef = max(history.history['val_dice_coef'])

# f = open(f'C:/Users/Mittal/Desktop/kunet/multikunet/output_with_weights.txt', "a")
# print("Max Dice Score: ", max_dice_coef, file=f)
# print("Max Val Dice Score: ", max_val_dice_coef, file=f)
# f.close()
    
# model.load_weights(f'C:/Users/Mittal/Desktop/kunet/multikunet/multi_kunet0_with_weights.h5')

# dice_scores = []

# for z in range(25):
#     test_img_number = random.randint(0, len(X_test)-1)
#     test_img = X_test[test_img_number]
#     ground_truth = y_test[test_img_number]
#     test_img_norm = test_img[:,:,0][:,:,None]
#     test_img_input = np.expand_dims(test_img_norm, 0)
#     prediction = (model.predict(test_img_input))
#     predicted_img = np.argmax(prediction, axis=3)[0,:,:]

#     # shape = prediction.shape
#     # prediction_dice = prediction

#     # for row in range(shape[0]): 
#     #     for col in range(shape[1]):
#     #         if (prediction[0, row, col, 0] <= 0.5):
#     #             prediction_dice[row, col] == 0

#     # prediction_dice = prediction_dice.astype(np.float32)
#     # ground_truth_dice = ground_truth.astype(np.float32)

#     # dice_score = dice_coef(ground_truth_dice, prediction_dice)
#     # dice_scores.append(dice_score)

#     # original_image_normalized = ground_truth.astype(float) / np.max(ground_truth)
#     # colored_mask = plt.get_cmap('jet')(prediction / np.max(prediction))
#     # alpha = 0.5 
#     # colored_mask[..., 3] = np.where(prediction > 0, alpha, 0)

#     plt.figure(figsize=(16, 8))
#     plt.subplot(131)
#     plt.title('Testing Image')
#     plt.imshow(test_img[:,:,0], cmap='gray')
#     plt.subplot(132)
#     plt.title('Testing Label')
#     plt.imshow(ground_truth[:,:,0], cmap='jet')
#     plt.subplot(133)
#     plt.title('Prediction on test image')
#     plt.imshow(predicted_img, cmap='jet')
#     # plt.subplot(144)
#     # plt.title("Overlayed Images")
#     # plt.imshow(original_image_normalized, cmap='jet')
#     # plt.imshow(colored_mask, cmap='jet')
#     plt.savefig(f'C:/Users/Mittal/Desktop/kunet/multikunet/predict/prediction_with_weights_{z}.png')
#     plt.close()

# f = open(f'C:/Users/Mittal/Desktop/kunet/multikunet/output_with_weights.txt', "a")
# print("Average Prediction Dice Score: ", np.mean(dice_scores), file=f)
# f.close()

# ##############################################################################################