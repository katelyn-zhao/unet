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
from IPython.display import Image, display
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, concatenate, UpSampling2D,BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.transform import resize
from sklearn.model_selection import KFold
plt.switch_backend('agg')

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
        BatchNormalization,
        Conv3D,
        Conv3DTranspose,
        MaxPooling3D,
        Dropout,
        SpatialDropout3D,
        UpSampling3D,
        Input,
        concatenate,
        multiply,
        add,
        Activation,
    )

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

data_path = 'C:/Users/mittal/Desktop/MRI'

import cv2

# change the tesing accordingly after changing batch_size
image_size = (256, 256)
batch_size = 8
epochs = 800


def resize_image(image, target_dims):
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


# get files from paths, no need to split train and test datasets here
# don't add '/' after the directory on the lab computer
train_images = sorted(glob(os.path.join('C:/Users/ester/Downloads/Segmentations/flint_adult_segmentations/Adult_fatF/fatfrac', '*.nii')))
train_masks = sorted(glob(os.path.join('C:/Users/ester/Downloads/Segmentations/flint_adult_segmentations/Adult_fatF/whole_liver_segmentation', '*.nii')))

image_dataset = []  
mask_dataset = []
sliced_image_dataset = []
sliced_mask_dataset = []

for i, image_name in enumerate(train_images):    
        image = nib.load(image_name)
        image = image.get_fdata()
        image = resize_image(image, image_size)
        image = np.array(image)
        image_dataset.append(np.array(image))

for i, image_name in enumerate(train_masks):
        image = nib.load(image_name)
        image = image.get_fdata()
        image = resize_image(image, image_size)
        image = np.array(image)
        mask_dataset.append(np.array(image))

for i in range(len(image_dataset)):
    for j in range(image_dataset[i].shape[2]):
        sliced_image_dataset.append(image_dataset[i][:,:,j])
        sliced_mask_dataset.append(mask_dataset[i][:,:,j])


sliced_image_dataset = np.expand_dims(np.array(sliced_image_dataset),3)
sliced_mask_dataset = np.expand_dims((np.array(sliced_mask_dataset)),3)

print("sliced image dataset: ", len(sliced_image_dataset))

train_steps = len(sliced_image_dataset)//batch_size
valid_steps = len(sliced_mask_dataset)//batch_size



import os
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x

def stem_block(x, n_filter, strides):
    x_init = x

    ## Conv 1
    x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same")(x)

    ## Shortcut
    s  = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
    s = BatchNormalization()(s)

    ## Add
    x = Add()([x, s])
    x = squeeze_excite_block(x)
    return x


def resnet_block(x, n_filter, strides=1):
    x_init = x

    ## Conv 1
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
    ## Conv 2
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same", strides=1)(x)

    ## Shortcut
    s  = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
    s = BatchNormalization()(s)

    ## Add
    x = Add()([x, s])
    x = squeeze_excite_block(x)
    return x

def aspp_block(x, num_filters, rate_scale=1):
    x1 = Conv2D(num_filters, (3, 3), dilation_rate=(6 * rate_scale, 6 * rate_scale), padding="same")(x)
    x1 = BatchNormalization()(x1)

    x2 = Conv2D(num_filters, (3, 3), dilation_rate=(12 * rate_scale, 12 * rate_scale), padding="same")(x)
    x2 = BatchNormalization()(x2)

    x3 = Conv2D(num_filters, (3, 3), dilation_rate=(18 * rate_scale, 18 * rate_scale), padding="same")(x)
    x3 = BatchNormalization()(x3)

    x4 = Conv2D(num_filters, (3, 3), padding="same")(x)
    x4 = BatchNormalization()(x4)

    y = Add()([x1, x2, x3, x4])
    y = Conv2D(num_filters, (1, 1), padding="same")(y)
    return y

def attetion_block(g, x):
    """
        g: Output of Parallel Encoder block
        x: Output of Previous Decoder block
    """

    filters = x.shape[-1]

    g_conv = BatchNormalization()(g)
    g_conv = Activation("relu")(g_conv)
    g_conv = Conv2D(filters, (3, 3), padding="same")(g_conv)

    g_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(g_conv)

    x_conv = BatchNormalization()(x)
    x_conv = Activation("relu")(x_conv)
    x_conv = Conv2D(filters, (3, 3), padding="same")(x_conv)

    gc_sum = Add()([g_pool, x_conv])

    gc_conv = BatchNormalization()(gc_sum)
    gc_conv = Activation("relu")(gc_conv)
    gc_conv = Conv2D(filters, (3, 3), padding="same")(gc_conv)

    gc_mul = Multiply()([gc_conv, x])
    return gc_mul

class ResUnetPlusPlus:
    def __init__(self, input_size=256):
        self.input_size = input_size

    def build_model(self):
        n_filters = [16, 32, 64, 128, 256]
        inputs = Input((256, 256, 1))

        c0 = inputs
        c1 = stem_block(c0, n_filters[0], strides=1)

        ## Encoder
        c2 = resnet_block(c1, n_filters[1], strides=2)
        c3 = resnet_block(c2, n_filters[2], strides=2)
        c4 = resnet_block(c3, n_filters[3], strides=2)

        ## Bridge
        b1 = aspp_block(c4, n_filters[4])

        ## Decoder
        d1 = attetion_block(c3, b1)
        d1 = UpSampling2D((2, 2))(d1)
        d1 = Concatenate()([d1, c3])
        d1 = resnet_block(d1, n_filters[3])

        d2 = attetion_block(c2, d1)
        d2 = UpSampling2D((2, 2))(d2)
        d2 = Concatenate()([d2, c2])
        d2 = resnet_block(d2, n_filters[2])

        d3 = attetion_block(c1, d2)
        d3 = UpSampling2D((2, 2))(d3)
        d3 = Concatenate()([d3, c1])
        d3 = resnet_block(d3, n_filters[1])

        ## output
        outputs = aspp_block(d3, n_filters[0])
        outputs = Conv2D(1, (1, 1), padding="same")(outputs)
        outputs = Activation("sigmoid")(outputs)

        ## Model
        model = Model(inputs, outputs)
        return model

def dice_coef(y_true, y_pred, smooth=1.):

    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


arch = ResUnetPlusPlus(input_size=image_size)
model = arch.build_model()


n_splits = 5

kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

dice_scores = []

# Iterate over each fold
for i, (train_index, test_index) in enumerate(kf.split(sliced_image_dataset, sliced_mask_dataset)):
    X_train, X_test = sliced_image_dataset[train_index], sliced_image_dataset[test_index]
    y_train, y_test = sliced_mask_dataset[train_index], sliced_mask_dataset[test_index]

    print("x-training: ", len(X_train))
    print("x-testing: ", len(X_test))
    print("y-training: ", len(y_train))
    print("y-testing: ", len(y_test))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='bce', metrics=[dice_coef])

    checkpoint = ModelCheckpoint(f'C:/Users/ester/Videos/resunet++/resunet++{i}.h5', monitor='val_loss', save_best_only=True)


    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        verbose = 1,
                        callbacks = [checkpoint])


    plt.figure(figsize=(12,3))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], color='r')
    plt.plot(history.history['val_loss'])
    plt.ylabel('BCE Losses')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val.'], loc='upper right')
    plt.subplot(1,2,2)
    plt.plot(history.history['dice_coef'], color='r')
    plt.plot(history.history['val_dice_coef'])
    plt.ylabel('Dice Score')
    plt.xlabel('Epoch')
    plt.tight_layout()
    #plt.show()
    plt.savefig(f'C:/Users/ester/Videos/resunet++/process{i}.png')
    plt.close()
    

    max_dice_coef = max(history.history['dice_coef'])

    f = open("C:/Users/ester/Videos/resunet++/output.txt", "a")
    print(max_dice_coef, file=f)
    f.close()
    
    
    model.load_weights(f'C:/Users/ester/Videos/resunet++/resunet++{i}.h5')

    # replace range with the number of files in the test set
    for z in range(len(X_test)):
        test_img = X_test[z]
        ground_truth = y_test[z]
        test_img_norm = test_img[:,:,0][:,:,None]
        test_img_input = np.expand_dims(test_img_norm, 0)
        prediction = (model.predict(test_img_input)[0,:,:,0] >= 0.5).astype(np.float64)

        original_image_normalized = ground_truth.astype(float) / np.max(ground_truth)
        colored_mask = plt.get_cmap('jet')(prediction / np.max(prediction))
        alpha = 0.5 
        colored_mask[..., 3] = np.where(prediction > 0, alpha, 0)

        dice_scores.append(dice_coef(ground_truth, prediction))

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
        plt.savefig(f'C:/Users/ester/Videos/resunet++/fold{i}_{z}.png')
        plt.close()
    # remove it to do 5 fold
    break
        
average_dice_coef = np.mean(dice_scores)

f = open(f'C:/Users/ester/Videos/resunet++/output.txt', "a")
print('average prediction dice score: ', average_dice_coef, file=f)
f.close()
