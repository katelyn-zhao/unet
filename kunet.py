import os
import io
import random
import nibabel
import cv2
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


def dice_coef(y_true, y_pred, smooth=1.):
  y_true_f = tf.keras.backend.flatten(y_true)
  y_pred_f = tf.keras.backend.flatten(y_pred)
  intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
  return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def tpr(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold)
   
    tp = np.sum((y_pred == 1) & (y_true == 1))

    fn = np.sum((y_pred == 0) & (y_true == 1))

    if (tp == 0):
        tpr = 0
    else:
        tpr = tp / (tp + fn)

    return tpr

def fpr(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold)

    fp = np.sum((y_pred == 1) & (y_true == 0))

    tn = np.sum((y_pred == 0) & (y_true == 0))
    
    if (fp == 0):
        fpr = 0
    else:
        fpr = fp / (fp + tn)

    return fpr


################################################################################################################################

def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c5 = BatchNormalization()(c5)
    p5 = MaxPooling2D(pool_size=(2,2))(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.3)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.3)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    c6 = BatchNormalization()(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    c7 = BatchNormalization()(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    c8 = BatchNormalization()(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    c9 = BatchNormalization()(c9)
     
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='bce', metrics=[dice_coef, tpr, fpr])
    model.summary()
    
    return model

################################################################################################################################

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
        angle = random.randint(10, 20)
        if cw:
            sliced_image_dataset.append(image_dataset[i][:,:,j].rotate(angle))
            sliced_mask_dataset.append(mask_dataset[i][:,:,j].rotate(angle))
        else:
            sliced_image_dataset.append(image_dataset[i][:,:,j].rotate(angle * -1))
            sliced_mask_dataset.append(mask_dataset[i][:,:,j].rotate(angle * -1))
        #contrast adjustment
        adjust = random.randint(0,1)
        if adjust:
            sliced_image_dataset.append(cv2.equalizeHist(image_dataset[i][:,:,j]))
            sliced_mask_dataset.append(cv2.equalizeHist(mask_dataset[i][:,:,j]))
        if adjust and cw:
            sliced_image_dataset.append(cv2.equalizeHist(image_dataset[i][:,:,j].rotate(angle)))
            sliced_mask_dataset.append(cv2.equalizeHist(mask_dataset[i][:,:,j].rotate(angle)))




#Normalize images
sliced_image_dataset = np.expand_dims(np.array(sliced_image_dataset),3)
#D not normalize masks, just rescale to 0 to 1.
sliced_mask_dataset = np.expand_dims((np.array(sliced_mask_dataset)),3)

# Sanity check, view a few images
# image_number = random.randint(0, len(X_train))
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(X_train[image_number], cmap='gray')
# plt.subplot(122)
# plt.imshow(y_train[image_number], cmap='gray')
# plt.show()

##############################################################################################################################################

#dice_scores = []
# TPRs = []
# FPRs = []

IMG_HEIGHT = sliced_image_dataset.shape[1]
IMG_WIDTH  = sliced_image_dataset.shape[2]
IMG_CHANNELS = sliced_image_dataset.shape[3]

def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

n_splits = 3

kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

# Iterate over each fold
for i, (train_index, test_index) in enumerate(kf.split(sliced_image_dataset, sliced_mask_dataset)):
    X_train, X_test = sliced_image_dataset[train_index], sliced_image_dataset[test_index]
    y_train, y_test = sliced_mask_dataset[train_index], sliced_mask_dataset[test_index]

    f = open("kunet/output.txt", "a")
    print("X-Training: ", len(X_train), file=f)
    print("X-Testing: ", len(X_test), file=f)
    print("Y-Training: ", len(y_train), file=f)
    print("Y-Testing: ", len(y_test), file=f)
    f.close()

    model = get_model()
   
    #checkpoint = ModelCheckpoint(f'kunet/best_model{i}.keras', monitor='val_loss', save_best_only=True)

    history = model.fit(X_train, y_train,
                        batch_size=16,
                        verbose=1,
                        epochs=1,
                        validation_data=(X_test, y_test),
                        shuffle=False)
                        #callbacks = [checkpoint]

    model.save(f'kunet/best_model{i}.keras')

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
    plt.savefig(f'kunet/process{i}.png')
    plt.close()

    max_dice_coef = max(history.history['dice_coef'])

    f = open("kunet/output.txt", "a")
    print(max_dice_coef, file=f)
    f.close()
    
    model.load_weights(f'kunet/best_model{i}.keras')

    for z in range(5):
        test_img_number = random.randint(0, len(X_test))
        test_img = X_test[test_img_number]
        ground_truth = y_test[test_img_number]
        test_img_norm = test_img[:,:,0][:,:,None]
        test_img_input = np.expand_dims(test_img_norm, 0)
        prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

        original_image_normalized = ground_truth.astype(float) / np.max(ground_truth)
        colored_mask = plt.get_cmap('jet')(prediction / np.max(prediction))
        alpha = 0.5 
        colored_mask[..., 3] = np.where(prediction > 0, alpha, 0)

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
        plt.savefig(f'kunet/predict/fold{i}_{z}.png')
        plt.close()
        
#average_dice_coef = np.mean(dice_scores)

# average_tpr = np.mean(TPRs)

# average_fpr = np.mean(FPRs)

# f = open("C:/katelynzhao/Desktop/kunet/output.txt", "a")
# #print('average prediction dice score', file=f)
# #print(average_dice_coef, file=f)
# print('average prediction tpr', file=f)
# print(average_tpr, file=f)
# print('average prediction fpr', file=f)
# print(average_fpr, file=f)
# f.close()

for i in range(n_splits):
    for j in range(10):
        patient_num = random.randint(0, len(image_dataset))
        anatomical = []
        labels = []
        predictions = []
        original = []
        colored_masks = []
        for k in range(10):
            slice_num = random.randint(0, image_dataset[patient_num].shape[2]-1)
            anatomical.append(image_dataset[patient_num][:,:,slice_num])

        for k in range(10):
            slice_num = random.randint(0, mask_dataset[patient_num].shape[2]-1)
            labels.append(mask_dataset[patient_num][:,:,slice_num])

        anatomical = np.expand_dims(np.array(anatomical),3)
        labels = np.expand_dims((np.array(labels)),3)

        for k in range(0, len(anatomical)):
            test_img = anatomical[k]
            ground_truth = labels[k]
            test_img_norm = test_img[:,:,0][:,:,None]
            test_img_input = np.expand_dims(test_img_norm, 0)
            prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
            predictions.append(prediction)

            original_image_normalized = ground_truth.astype(float) / np.max(ground_truth)
            original.append(original_image_normalized)
            colored_mask = plt.get_cmap('jet')(prediction / np.max(prediction))
            alpha = 0.5 
            colored_mask[..., 3] = np.where(prediction > 0, alpha, 0)
            colored_masks.append(colored_mask)
            
        
        plt.figure(figsize=(16, 20))
        plt.subplot(5,4,1)
        plt.title('Testing Image')
        plt.imshow(anatomical[0][:,:,0], cmap='gray')
        plt.subplot(5,4,2)
        plt.title('Testing Label')
        plt.imshow(labels[0][:,:,0], cmap='gray')
        plt.subplot(5,4,3)
        plt.title('Prediction on test image')
        plt.imshow(predictions[0], cmap='gray')
        plt.subplot(5,4,4)
        plt.title("Overlayed Images")
        plt.imshow(original[0], cmap='gray')
        plt.imshow(colored_masks[0], cmap='jet')
        plt.subplot(5,4,5)
        plt.title('Testing Image')
        plt.imshow(anatomical[1][:,:,0], cmap='gray')
        plt.subplot(5,4,6)
        plt.title('Testing Label')
        plt.imshow(labels[1][:,:,0], cmap='gray')
        plt.subplot(5,4,7)
        plt.title('Prediction on test image')
        plt.imshow(predictions[1], cmap='gray')
        plt.subplot(5,4,8)
        plt.title("Overlayed Images")
        plt.imshow(original[1], cmap='gray')
        plt.imshow(colored_masks[1], cmap='jet')
        plt.subplot(5,4,9)
        plt.title('Testing Image')
        plt.imshow(anatomical[2][:,:,0], cmap='gray')
        plt.subplot(5,4,10)
        plt.title('Testing Label')
        plt.imshow(labels[2][:,:,0], cmap='gray')
        plt.subplot(5,4,11)
        plt.title('Prediction on test image')
        plt.imshow(predictions[2], cmap='gray')
        plt.subplot(5,4,12)
        plt.title("Overlayed Images")
        plt.imshow(original[2], cmap='gray')
        plt.imshow(colored_masks[2], cmap='jet')
        plt.subplot(5,4,13)
        plt.title('Testing Image')
        plt.imshow(anatomical[3][:,:,0], cmap='gray')
        plt.subplot(5,4,14)
        plt.title('Testing Label')
        plt.imshow(labels[3][:,:,0], cmap='gray')
        plt.subplot(5,4,15)
        plt.title('Prediction on test image')
        plt.imshow(predictions[3], cmap='gray')
        plt.subplot(5,4,16)
        plt.title("Overlayed Images")
        plt.imshow(original[3], cmap='gray')
        plt.imshow(colored_masks[3], cmap='jet')
        plt.subplot(5,4,17)
        plt.title('Testing Image')
        plt.imshow(anatomical[4][:,:,0], cmap='gray')
        plt.subplot(5,4,18)
        plt.title('Testing Label')
        plt.imshow(labels[4][:,:,0], cmap='gray')
        plt.subplot(5,4,19)
        plt.title('Prediction on test image')
        plt.imshow(predictions[4], cmap='gray')
        plt.subplot(5,4,20)
        plt.title("Overlayed Images")
        plt.imshow(original[4], cmap='gray')
        plt.imshow(colored_masks[4], cmap='jet')
        plt.savefig(f'kunet/predict/patient{j}_01234.png')
        plt.close()

        plt.figure(figsize=(16, 20))
        plt.subplot(5,4,1)
        plt.title('Testing Image')
        plt.imshow(anatomical[5][:,:,0], cmap='gray')
        plt.subplot(5,4,2)
        plt.title('Testing Label')
        plt.imshow(labels[5][:,:,0], cmap='gray')
        plt.subplot(5,4,3)
        plt.title('Prediction on test image')
        plt.imshow(predictions[5], cmap='gray')
        plt.subplot(5,4,4)
        plt.title("Overlayed Images")
        plt.imshow(original[5], cmap='gray')
        plt.imshow(colored_masks[5], cmap='jet')
        plt.subplot(5,4,5)
        plt.title('Testing Image')
        plt.imshow(anatomical[6][:,:,0], cmap='gray')
        plt.subplot(5,4,6)
        plt.title('Testing Label')
        plt.imshow(labels[6][:,:,0], cmap='gray')
        plt.subplot(5,4,7)
        plt.title('Prediction on test image')
        plt.imshow(predictions[6], cmap='gray')
        plt.subplot(5,4,8)
        plt.title("Overlayed Images")
        plt.imshow(original[6], cmap='gray')
        plt.imshow(colored_masks[6], cmap='jet')
        plt.subplot(5,4,9)
        plt.title('Testing Image')
        plt.imshow(anatomical[7][:,:,0], cmap='gray')
        plt.subplot(5,4,10)
        plt.title('Testing Label')
        plt.imshow(labels[7][:,:,0], cmap='gray')
        plt.subplot(5,4,11)
        plt.title('Prediction on test image')
        plt.imshow(predictions[7], cmap='gray')
        plt.subplot(5,4,12)
        plt.title("Overlayed Images")
        plt.imshow(original[7], cmap='gray')
        plt.imshow(colored_masks[7], cmap='jet')
        plt.subplot(5,4,13)
        plt.title('Testing Image')
        plt.imshow(anatomical[8][:,:,0], cmap='gray')
        plt.subplot(5,4,14)
        plt.title('Testing Label')
        plt.imshow(labels[8][:,:,0], cmap='gray')
        plt.subplot(5,4,15)
        plt.title('Prediction on test image')
        plt.imshow(predictions[8], cmap='gray')
        plt.subplot(5,4,16)
        plt.title("Overlayed Images")
        plt.imshow(original[8], cmap='gray')
        plt.imshow(colored_masks[8], cmap='jet')
        plt.subplot(5,4,17)
        plt.title('Testing Image')
        plt.imshow(anatomical[9][:,:,0], cmap='gray')
        plt.subplot(5,4,18)
        plt.title('Testing Label')
        plt.imshow(labels[9][:,:,0], cmap='gray')
        plt.subplot(5,4,19)
        plt.title('Prediction on test image')
        plt.imshow(predictions[9], cmap='gray')
        plt.subplot(5,4,20)
        plt.title("Overlayed Images")
        plt.imshow(original[9], cmap='gray')
        plt.imshow(colored_masks[9], cmap='jet')
        plt.savefig(f'kunet/predict/patient{j}_56789.png')
        plt.close()


