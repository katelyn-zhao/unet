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
    optimizer = Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[dice_coef])
    model.summary()

    return model


segmentation_folder = 'C:/Users/Mittal/Desktop/MutliUnet_data/combined/'
ff_folder = 'C:/Users/Mittal/Desktop/MutliUnet_data/ff/'
r2_folder = 'C:/Users/Mittal/Desktop/MutliUnet_data/r2/'
water_folder = 'C:/Users/Mittal/Desktop/MutliUnet_data/water/'

ff_images = []
r2_images = []
water_images = []
images = []
masks = []
sliced_image_dataset = []
sliced_mask_dataset = []

def resize_image(image, target_dims=(256, 256)):
    rows, cols = image.shape[:2]
    target_rows, target_cols = target_dims

    pad_vert = target_rows - rows
    pad_top = pad_vert // 2
    pad_bot = pad_vert - pad_top

    pad_horz = target_cols - cols
    pad_left = pad_horz // 2
    pad_right = pad_horz - pad_left

    img_padded = cv2.copyMakeBorder(image, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_CONSTANT)
    return img_padded


def normalize_image(image):
    image = image - np.min(image)
    image = image / np.max(image)
    return image

ff_image = sorted(os.listdir(ff_folder))
for i, image_name in enumerate(ff_image):
    if (image_name.split('.')[1] == 'nii'):
        image = nib.load(ff_folder+image_name)
        image = np.array(image.get_fdata())
        image = resize_image(image)
        image = normalize_image(image)
        ff_images.append(np.array(image))

r2_image = sorted(os.listdir(r2_folder))
for i, image_name in enumerate(ff_image):
    if (image_name.split('.')[1] == 'nii'):
        image = nib.load(r2_folder+image_name)
        image = np.array(image.get_fdata())
        image = resize_image(image)
        image = normalize_image(image)
        r2_images.append(np.array(image))

water_image = sorted(os.listdir(water_folder))
for i, image_name in enumerate(ff_image):
    if (image_name.split('.')[1] == 'nii'):
        image = nib.load(water_folder+image_name)
        image = np.array(image.get_fdata())
        image = resize_image(image)
        image = normalize_image(image)
        water_images.append(np.array(image))

for i in range(len(ff_images)):
    whole_image = np.stack((ff_images[i], r2_images[i], water_images[i]), axis=-1)
    images.append(whole_image)

mask_image = sorted(os.listdir(segmentation_folder))
for i, image_name in enumerate(mask_image):
    if (image_name.split('.')[1] == 'nii'):
        image = nib.load(segmentation_folder+image_name)
        image = np.array(image.get_fdata())
        image = resize_image(image)
        image = normalize_image(image)
        masks.append(np.array(image))

for i in range(len(images)):
    if images[i].shape[2] <= masks[i].shape[2]:
        for j in range(images[i].shape[2]):
            sliced_image_dataset.append(images[i][:,:,j,:])
            sliced_mask_dataset.append(masks[i][:,:,j])
    else:
        for j in range(masks[i].shape[2]):
            sliced_image_dataset.append(images[i][:,:,j,:])
            sliced_mask_dataset.append(masks[i][:,:,j])

sliced_image_dataset = np.array(sliced_image_dataset)[..., np.newaxis]
sliced_mask_dataset = np.array(sliced_mask_dataset)[..., np.newaxis]

f = open(f"C:/Users/Mittal/Desktop/kunet/model7_output.txt", "a")
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

    f = open(f"C:/Users/Mittal/Desktop/kunet/model7_output.txt", "a")
    print("FOLD----------------------------------", file=f)
    print("x-training: ", len(X_train), file=f)
    print("x-testing: ", len(X_test), file=f)
    print("y-training: ", len(y_train), file=f)
    print("y-testing: ", len(y_test), file=f)
    f.close()

    model = get_model()

    checkpoint = ModelCheckpoint(f'C:/Users/Mittal/Desktop/kunet/model7_{i}.h5', monitor='val_loss', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        batch_size=16,
                        verbose=1,
                        epochs=1000,
                        validation_data=(X_test, y_test),
                        shuffle=False,
                        callbacks=[checkpoint, early_stopping])

    model_save_path = f'C:/Users/Mittal/Desktop/kunet/finalmodel7_{i}.h5'

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
    plt.savefig(f'C:/Users/Mittal/Desktop/kunet/model7_process{i}.png')
    plt.close()

    max_dice_coef = max(history.history['dice_coef'])
    max_val_dice_coef = max(history.history['val_dice_coef'])
    # max_tpr = max(history.history['tpr'])
    # min_fpr = min(history.history['fpr'])

    f = open(f"C:/Users/Mittal/Desktop/kunet/model7_output.txt", "a")
    print("max dice coef: ", max_dice_coef, file=f)
    print("max val dice coef: ", max_val_dice_coef, file=f)
    # print("max tpr: ", max_tpr, file=f)
    # print("min fpr: ", min_fpr, file=f)
    f.close()