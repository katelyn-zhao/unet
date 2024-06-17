import os
import io
import random
import nibabel
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
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

def tprf(y_true, y_pred, threshold=0.5):
    y_pred_pos = tf.cast(y_pred > threshold, tf.float32)
    y_true_pos = tf.cast(y_true > threshold, tf.float32)
    
    true_pos = tf.reduce_sum(tf.cast(tf.logical_and(y_true_pos == 1, y_pred_pos == 1), tf.float32))
    actual_pos = tf.reduce_sum(tf.cast(y_true_pos, tf.float32))
    
    tpr = true_pos / (actual_pos + tf.keras.backend.epsilon())
    return tpr

def fprf(y_true, y_pred, threshold=0.5):
    y_pred_pos = tf.cast(y_pred > threshold, tf.float32)
    y_true_neg = tf.cast(y_true <= threshold, tf.float32)
    
    false_pos = tf.reduce_sum(tf.cast(tf.logical_and(y_true_neg == 1, y_pred_pos == 1), tf.float32))
    actual_neg = tf.reduce_sum(tf.cast(y_true_neg, tf.float32))
    
    fpr = false_pos / (actual_neg + tf.keras.backend.epsilon())
    return fpr

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
    u7 = concatenate([u7, att_7, Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)])
    c7 = conv_block(u7, 64)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    att_8 = attention_gate(u8, c2, 16)
    u8 = concatenate([u8, att_8, Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c3), Conv2DTranspose(32, (3, 3), strides=(4, 4), padding='same')(c4)])
    c8 = conv_block(u8, 32)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    att_9 = attention_gate(u9, c1, 16)
    u9 = concatenate([u9, att_9, Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c2), Conv2DTranspose(16, (3, 3), strides=(4, 4), padding='same')(c3), Conv2DTranspose(16, (4, 4), strides=(8, 8), padding='same')(c4)])
    c9 = conv_block(u9, 16)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    optimizer = Adam(learning_rate=1e-3)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef, tprf, fprf])
    model.summary()

    return model

image_directory = 'C:/Users/Mittal/Desktop/wunet++/model_1/fatfrac_combined/'
mask_directory = 'C:/Users/Mittal/Desktop/wunet++/model_1/whole_liver_segmentation_combined/'

image_dataset = []  
mask_dataset = []
sliced_image_dataset = []
sliced_mask_dataset = []

images = os.listdir(image_directory)
for i, image_name in enumerate(images):    
    if (image_name.split('.')[1] == 'nii'):
        image = nibabel.load(image_directory + image_name)
        image = np.array(image.get_fdata())
        image = preprocess_volume(image)  # Preprocess the image
        image_dataset.append(np.array(image))

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'nii'):
        image = nibabel.load(mask_directory + image_name)
        image = np.array(image.get_fdata())
        image = preprocess_volume(image)  # Preprocess the mask
        mask_dataset.append(np.array(image))

sliced_image_filenames = []

for i, image_name in enumerate(images):
    if (image_name.split('.')[1] == 'nii'):
        sliced_image_filenames.append(os.path.splitext(image_name)[0])  # Extract filename without extension

for i in range(len(image_dataset)):
    num_slices = image_dataset[i].shape[2]
    for j in range(num_slices):
        sliced_image_dataset.append(image_dataset[i][:,:,j])

for i in range(len(mask_dataset)):
    num_slices = mask_dataset[i].shape[2]
    for j in range(num_slices):
        sliced_mask_dataset.append(mask_dataset[i][:,:,j])

# Normalize and convert images
sliced_image_dataset = np.expand_dims(np.array(sliced_image_dataset), 3)
sliced_image_dataset = sliced_image_dataset.astype('float64')

# Convert masks and ensure they are in the correct format
sliced_mask_dataset = np.expand_dims(np.array(sliced_mask_dataset), 3)
sliced_mask_dataset = sliced_mask_dataset.astype('float64')

# Split the dataset into training and testing sets

print("Number of images:", len(sliced_image_dataset))
print("Number of masks:", len(sliced_mask_dataset))

if len(sliced_image_dataset) != len(sliced_mask_dataset):
    print("Warning: The datasets do not have the same length.")
    # Optionally, align them by removing the extra entry
    min_length = min(len(sliced_image_dataset), len(sliced_mask_dataset))
    sliced_image_dataset = sliced_image_dataset[:min_length]
    sliced_mask_dataset = sliced_mask_dataset[:min_length]


# X_train, X_test, y_train, y_test = train_test_split(sliced_image_dataset, sliced_mask_dataset, test_size = 0.20, random_state = 42)

dice_scores = []
TPRs = []
FPRs = []

IMG_HEIGHT = sliced_image_dataset.shape[1]
IMG_WIDTH  = sliced_image_dataset.shape[2]
IMG_CHANNELS = sliced_image_dataset.shape[3]

def get_model():
    return simple_unet_plus_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

n_splits = 5

kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

histories = []

for i, (train_index, test_index) in enumerate(kf.split(sliced_image_dataset, sliced_mask_dataset)):
    X_train, X_test = sliced_image_dataset[train_index], sliced_image_dataset[test_index]
    y_train, y_test = sliced_mask_dataset[train_index], sliced_mask_dataset[test_index]

    model = get_model()

    checkpoint = ModelCheckpoint(f'C:/Users/Mittal/Desktop/wunet++/256pad/test_best_model{i}.h5', monitor='val_loss', save_best_only=True)

    history = model.fit(X_train, y_train,
                        batch_size=16,
                        verbose=1,
                        epochs=800,
                        validation_data=(X_test, y_test),
                        shuffle=False,
                        callbacks=[checkpoint])

    histories.append(history)

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
    plt.savefig(f'C:/Users/Mittal/Desktop/wunet++/256pad/process{i}.png')
    plt.close()

    max_dice_coef = max(history.history['dice_coef'])
    max_val_dice_coef = max(history.history['val_dice_coef'])
    max_tpr = max(history.history['tprf'])
    min_fpr = min(history.history['fprf'])

    with open("C:/Users/Mittal/Desktop/wunet++/256pad/output.txt", "a") as f:
        print("FOLD: ", {i}, file=f)
        print("max training dice: ", max_dice_coef, file=f)
        print("max val dice: ", max_val_dice_coef, file=f)
        print("max tpr: ", max_tpr,file=f)
        print("min fpr: ", min_fpr, file=f)
    f.close()

    model.load_weights(f'C:/Users/Mittal/Desktop/wunet++/256pad/test_best_model{i}.h5')

    # Ensure all slices for each file are processed
    filewise_predictions = {filename: [] for filename in sliced_image_filenames}
    for idx, filename in enumerate(sliced_image_filenames):
        num_slices_per_image = image_dataset[idx].shape[2]  # Assuming image_dataset is accessible here
        start_index = sum(image_dataset[i].shape[2] for i in range(idx))  # Start index for slices of this image
        for z in range(num_slices_per_image):
            test_img = sliced_image_dataset[start_index + z]
            ground_truth = sliced_mask_dataset[start_index + z]
            test_img_norm = test_img[:, :, 0][:, :, None]
            test_img_input = np.expand_dims(test_img, axis=0)
            prediction = (model.predict(test_img_input)[0, :, :, 0] > 0.5).astype(np.uint8)
            filewise_predictions[filename].append(prediction)

            # Save the slice-by-slice visualization
            original_image_normalized = ground_truth.astype(float) / np.max(ground_truth)
            colored_mask = plt.get_cmap('jet')(prediction / np.max(prediction))
            alpha = 0.5
            colored_mask[..., 3] = np.where(prediction > 0, alpha, 0)

            plt.figure(figsize=(16, 8))
            plt.subplot(141)
            plt.title('Anatomical Image')
            plt.imshow(test_img[:, :, 0], cmap='gray')
            plt.subplot(142)
            plt.title('Actual Mask')
            plt.imshow(ground_truth[:, :, 0], cmap='gray')
            plt.subplot(143)
            plt.title('Prediction Mask')
            plt.imshow(prediction, cmap='gray')
            plt.subplot(144)
            plt.title("Overlayed Images")
            plt.imshow(original_image_normalized, cmap='gray')
            plt.imshow(colored_mask, cmap='jet')
            plt.savefig(f'C:/Users/Mittal/Desktop/wunet++/256pad/predict/fold{i}_{filename}_slice{z}.png')
            plt.close()

    # Saving 3D predictions
    for filename, predictions in filewise_predictions.items():
        if len(predictions) > 0:  # Check if predictions are available
            three_d_predictions_volume = np.stack(predictions, axis=-1)
            affine = np.eye(4)
            nii_img = nibabel.Nifti1Image(three_d_predictions_volume, affine)
            nibabel.save(nii_img, f'C:/Users/Mittal/Desktop/wunet++/256pad/3dpredict/fold{i}_{filename}_3d_predictions.nii')
        else:
            print(f"No predictions available for {filename}.")
