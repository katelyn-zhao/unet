import os
import io
import random
import nibabel as nib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import cv2
from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from sklearn.model_selection import KFold
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda

def preprocess_volume(volume):
    pad_x = max(0, 256 - volume.shape[0])
    pad_y = max(0, 256 - volume.shape[1])
    pad_z = max(0, 40 - volume.shape[2])
    pad_width = ((0, pad_x), (0, pad_y), (0, pad_z))
    volume_padded = np.pad(volume, pad_width, mode='constant', constant_values=0.0)
    return volume_padded

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = tf.cast(tf.keras.backend.flatten(y_true), tf.float64)
    y_pred_f = tf.cast(tf.keras.backend.flatten(y_pred), tf.float64)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def conv_block(input_tensor, num_filters):
    x = Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(input_tensor)
    x = Dropout(0.1)(x)
    x = Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    return x

def attention_gate(inp_1, inp_2, num_filters):
    g = Conv2D(num_filters, (1, 1), kernel_initializer='he_normal')(inp_1)
    x = Conv2D(num_filters, (1, 1), kernel_initializer='he_normal')(inp_2)
    x = concatenate([g, x])
    x = Conv2D(1, (1, 1), kernel_initializer='he_normal', activation='sigmoid')(x)
    return x

def simple_unet_plus_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    # Contraction path (Encoder)
    c1 = conv_block(s, 16)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = conv_block(p1, 32)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = conv_block(p2, 64)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = conv_block(p3, 128)
    p4 = MaxPooling2D((2, 2))(c4)
    c5 = conv_block(p4, 256)

    # Expansive path with attention gates (Decoder)
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    att_6 = attention_gate(u6, c4, 64)
    u6 = concatenate([u6, att_6])
    c6 = conv_block(u6, 128)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    att_7 = attention_gate(u7, c3, 32)
    u7 = concatenate([u7, att_7])
    c7 = conv_block(u7, 64)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    att_8 = attention_gate(u8, c2, 16)
    u8 = concatenate([u8, att_8])
    c8 = conv_block(u8, 32)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    att_9 = attention_gate(u9, c1, 16)
    u9 = concatenate([u9, att_9])
    c9 = conv_block(u9, 16)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    # optimizer = Adam(learning_rate=1e-3)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
    model.summary()

    return model


image_directory = 'C:/Users/Mittal/Desktop/MutliUnet_data/water/'
mask_directory = 'C:/Users/Mittal/Desktop/MutliUnet_data/combined/'

image_dataset = []  
mask_dataset = []
sliced_image_dataset = []
sliced_mask_dataset = []

def pad_volume(volume):
    pad_x = max(0, 256 - volume.shape[0])
    pad_y = max(0, 256 - volume.shape[1])
    pad_x_begin = pad_x // 2
    pad_x_end = pad_x - pad_x_begin
    pad_y_begin = pad_y // 2
    pad_y_end = pad_y - pad_y_begin
    pad_width = ((pad_x_begin, pad_x_end), (pad_y_begin, pad_y_end), (0, 0))
    volume_padded = np.pad(volume, pad_width, mode='constant', constant_values=0.0)
    return volume_padded

def normalize_image(image):
    image = image - np.min(image)
    image = image / np.max(image)
    return image

images = sorted(os.listdir(image_directory))
for i, image_name in enumerate(images):    
    if (image_name.split('.')[1] == 'nii'):
        image = nib.load(image_directory+image_name)
        image = np.array(image.get_fdata())
        image = pad_volume(image)
        image = normalize_image(image)
        image_dataset.append(np.array(image))

masks = sorted(os.listdir(mask_directory))
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'nii'):
        image = nib.load(mask_directory+image_name)
        image = np.array(image.get_fdata())
        image = pad_volume(image)
        mask_dataset.append(np.array(image))

for i in range(len(image_dataset)):
    for j in range(image_dataset[i].shape[2]):
        sliced_image_dataset.append(image_dataset[i][:,:,j])
        sliced_mask_dataset.append(mask_dataset[i][:,:,j])

sliced_image_dataset = np.array(sliced_image_dataset)
sliced_mask_dataset = np.array(sliced_mask_dataset)

sliced_image_dataset = np.expand_dims(sliced_image_dataset, axis=3)
sliced_mask_dataset = np.expand_dims(sliced_mask_dataset, axis=3)

f = open(f"C:/Users/Mittal/Desktop/kunet/model10_output.txt", "a")
print("sliced image dataset: ", len(sliced_image_dataset), file=f)
f.close()

IMG_HEIGHT = sliced_image_dataset.shape[1]
IMG_WIDTH  = sliced_image_dataset.shape[2]
IMG_CHANNELS = sliced_image_dataset.shape[3]

def get_model():
    return simple_unet_plus_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

n_splits = 5

kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

for i, (train_index, test_index) in enumerate(kf.split(sliced_image_dataset, sliced_mask_dataset)):
    X_train, X_test = sliced_image_dataset[train_index], sliced_image_dataset[test_index]
    y_train, y_test = sliced_mask_dataset[train_index], sliced_mask_dataset[test_index]

    f = open(f"C:/Users/Mittal/Desktop/kunet/model10_output.txt", "a")
    print("FOLD----------------------------------", file=f)
    print("x-training: ", len(X_train), file=f)
    print("x-testing: ", len(X_test), file=f)
    print("y-training: ", len(y_train), file=f)
    print("y-testing: ", len(y_test), file=f)
    f.close()

    model = get_model()

    checkpoint = ModelCheckpoint(f'C:/Users/Mittal/Desktop/kunet/model10_{i}.h5', monitor='val_loss', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        batch_size=16,
                        verbose=1,
                        epochs=1000,
                        validation_data=(X_test, y_test),
                        shuffle=False,
                        callbacks=[checkpoint, early_stopping])

    model_save_path = f'C:/Users/Mittal/Desktop/kunet/finalmodel10_{i}.h5'

    model.save(model_save_path)
    print(f'Model for fold {i} saved to {model_save_path}')

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], color='r')
    plt.plot(history.history['val_loss'])
    plt.ylabel('Losses')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val.'], loc='upper right')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['dice_coef'], color='r')
    plt.plot(history.history['val_dice_coef'])
    plt.ylabel('dice_coef')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(f'C:/Users/Mittal/Desktop/kunet/model10_process{i}.png')
    plt.close()

    max_dice_coef = max(history.history['dice_coef'])
    max_val_dice_coef = max(history.history['val_dice_coef'])
    # max_tpr = max(history.history['tpr'])
    # min_fpr = min(history.history['fpr'])

    f = open(f"C:/Users/Mittal/Desktop/kunet/model10_output.txt", "a")
    print("max dice coef: ", max_dice_coef, file=f)
    print("max val dice coef: ", max_val_dice_coef, file=f)
    # print("max tpr: ", max_tpr, file=f)
    # print("min fpr: ", min_fpr, file=f)
    f.close()