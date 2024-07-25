import nibabel as nib
import numpy as np
import tensorflow as tf
import cv2
from glob import glob
import os
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Conv2DTranspose, MaxPooling2D, Concatenate, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split



data_path = 'C:/Users/ester/Downloads/Segmentations/flint_adult_segmentations/Adult_fatF'
batch_size = 1
image_size = (256, 256)

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


class NiiDataGenerator(Sequence):
    def __init__(self, image_filenames, mask_filenames, batch_size, image_size):
        self.image_filenames = image_filenames
        self.mask_filenames = mask_filenames
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.mask_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        x = np.zeros((self.batch_size, *self.image_size, 1))
        y = np.zeros((self.batch_size, *self.image_size, 1))

        for i, (image_filename, mask_filename) in enumerate(zip(batch_x, batch_y)):
            image = nib.load(image_filename)
            mask = nib.load(mask_filename)

            image_data = image.get_fdata()
            mask_data = mask.get_fdata()

            image_data = resize_image(image_data, image_size)
            mask_data = resize_image(mask_data, image_size)   

            slice_index = 7  # Example slice index

            x[i, :, :, 0] = image_data[:, :, slice_index]
            y[i, :, :, 0] = mask_data[:, :, slice_index]
        return x, y



train_images = sorted(glob(os.path.join(data_path, 'fatfrac', 'f_*.nii')) )
train_masks = sorted(glob(os.path.join(data_path, 'whole_liver_segmentation', 'f_*.nii')))


                     
print(f'Number of images: {len(train_images)}')
print(f'Number of masks: {len(train_masks)}')



                     
train_image_files, val_image_files, train_mask_files, val_mask_files = train_test_split(
    train_images, train_masks, test_size=0.2, random_state=42
)

train_gen = NiiDataGenerator(train_image_files, train_mask_files, batch_size, image_size)
val_gen = NiiDataGenerator(val_image_files, val_mask_files, batch_size, image_size)


def conv_block(x, filters, kernel_size=3, padding='same'):
    x = Conv2D(filters, kernel_size=kernel_size, padding=padding)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def atrous_conv_block(x, filters, rate=1):
    x = Conv2D(filters, kernel_size=3, padding='same', dilation_rate=rate)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def aspp_block(x, filters):
    shape = x.shape[1:3]
    y1 = AveragePooling2D(pool_size=(shape[0], shape[1]))(x)
    y1 = Conv2D(filters, 1, padding="same")(y1)
    y1 = BatchNormalization()(y1)
    y1 = ReLU()(y1)
    y1 = tf.image.resize(y1, (shape[0], shape[1]))

    y2 = Conv2D(filters, 1, dilation_rate=1, padding="same")(x)
    y2 = BatchNormalization()(y2)
    y2 = ReLU()(y2)

    y3 = Conv2D(filters, 3, dilation_rate=6, padding="same")(x)
    y3 = BatchNormalization()(y3)
    y3 = ReLU()(y3)

    y4 = Conv2D(filters, 3, dilation_rate=12, padding="same")(x)
    y4 = BatchNormalization()(y4)
    y4 = ReLU()(y4)

    y5 = Conv2D(filters, 3, dilation_rate=18, padding="same")(x)
    y5 = BatchNormalization()(y5)
    y5 = ReLU()(y5)

    y = Concatenate()([y1, y2, y3, y4, y5])

    y = Conv2D(filters, 1, padding="same")(y)
    y = BatchNormalization()(y)
    y = ReLU()(y)

    return y

def deep_lab_v3_plus(input_shape):
    inputs = Input(shape=input_shape)

    # Encoder with Xception or MobileNetV2 backbone
    # For simplicity, using custom layers here
    x = conv_block(inputs, 64, kernel_size=7)
    x = MaxPooling2D(pool_size=2, strides=2)(x)

    x = atrous_conv_block(x, 128, rate=1)
    x = atrous_conv_block(x, 256, rate=2)
    x = atrous_conv_block(x, 512, rate=4)

    # ASPP
    x = aspp_block(x, 256)

    # Decoder
    x = Conv2DTranspose(256, kernel_size=3, strides=2, padding='same')(x)
    x = conv_block(x, 256)

    x = Conv2D(1, kernel_size=1, activation='sigmoid')(x)

    model = Model(inputs, x)
    return model

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


input_shape = (256, 256, 1)
model = deep_lab_v3_plus(input_shape)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
model.summary()
checkpoint = ModelCheckpoint('best_modelD.h5', monitor='val_dice_coef', save_best_only=True, mode='max')

# Train the model with validation data
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=[checkpoint]
)


model.save('deep_lab_v3_plus_model.h5')


def predict(input_data):
    input_data = np.expand_dims(resize_image(input_data, image_size), axis=0)
    prediction = model.predict(input_data)
    return prediction

##
### Example prediction
##example_data = nib.load('C:/Users/ester/Downloads/Segmentations/flint_adult_segmentations/Adult_fatF/fatfrac/f_2396.nii').get_fdata()
##prediction = predict(example_data)
##print(prediction)
