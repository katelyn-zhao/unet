import os
import random
import nibabel as nib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from glob import glob
from sklearn.model_selection import KFold
import os
import io
import random
import nibabel
import numpy as np
from glob import glob
import nibabel as nib
import tensorflow as tf
from nibabel import load
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.utils import Sequence
from IPython.display import Image, display
from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras import layers
from keras.layers import Input, concatenate, UpSampling2D,BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.transform import rotate
from skimage.util import montage
# Paths to your data
segmentation_folder = 'C:/Users/ester/Downloads/OneDrive_2024-08-16/LIG-AZifan Shared Folder/GEA21_Segmentation_Corrections_For_Ali/GEA21_Segmentation_Corrections_For_Ali/combined/combined'
modality_folders = {
    'art': 'C:/Users/ester/Downloads/OneDrive_2024-08-16/LIG-AZifan Shared Folder/GEA21_Segmentation_Corrections_For_Ali/GEA21_Segmentation_Corrections_For_Ali/combined/water',
    'ven': 'C:/Users/ester/Downloads/OneDrive_2024-08-16/LIG-AZifan Shared Folder/GEA21_Segmentation_Corrections_For_Ali/GEA21_Segmentation_Corrections_For_Ali/combined/r2',
    'pre': 'C:/Users/ester/Downloads/OneDrive_2024-08-16/LIG-AZifan Shared Folder/GEA21_Segmentation_Corrections_For_Ali/GEA21_Segmentation_Corrections_For_Ali/combined/ff'
}

def get_base_filename(filepath):
    """Extract base filename without extension."""
    return os.path.splitext(os.path.basename(filepath))[0]

def check_file_matches(segmentation_files, modality_folders):
    missing_files = []
    
    # List all files in modality folders
    modality_files = {key: set(get_base_filename(f) for f in glob(os.path.join(path, '*.nii*'))) for key, path in modality_folders.items()}
    
    for seg_file in segmentation_files:
        base_filename = get_base_filename(seg_file)
        
        
        if not all(base_filename in modality_files[key] for key in modality_folders):
            missing_files.append(base_filename)
    
    return missing_files


segmentation_files = glob(os.path.join(segmentation_folder, '*.nii*'))


missing_files = check_file_matches(segmentation_files, modality_folders)


if missing_files:
    print("Files with missing matches:")
    for file in missing_files:
        print(file)
else:
    print("All files have corresponding matches in the modality folders.")




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


def normalize_image(image):
    image = image - np.min(image)
    image = image / np.max(image)
    return image


class NiiDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, segmentation_files, batch_size, image_size):
        self.segmentation_files = segmentation_files
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return int(np.ceil(len(self.segmentation_files) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_seg_files = self.segmentation_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        x = np.zeros((self.batch_size, *self.image_size, 3)) 
        y = np.zeros((self.batch_size, *self.image_size, 1))
        
        for i, seg_file in enumerate(batch_seg_files):
            base_filename = os.path.basename(seg_file).replace('_seg', '')

        
            art_file = os.path.join(modality_folders['art'], base_filename)
            ven_file = os.path.join(modality_folders['ven'], base_filename)
            pre_file = os.path.join(modality_folders['pre'], base_filename)

     
            if not (os.path.exists(art_file) and os.path.exists(ven_file) and os.path.exists(pre_file)):
                print(f"Skipping {base_filename} due to missing modality file.")
                continue

     
            art_data = nib.load(art_file).get_fdata()
            ven_data = nib.load(ven_file).get_fdata()
            pre_data = nib.load(pre_file).get_fdata()
            mask_data = nib.load(seg_file).get_fdata()

         
            art_data = resize_image(art_data, self.image_size)
            ven_data = resize_image(ven_data, self.image_size)
            pre_data = resize_image(pre_data, self.image_size)
            mask_data = resize_image(mask_data, self.image_size)

        
            art_data = normalize_image(art_data)
            ven_data = normalize_image(ven_data)
            pre_data = normalize_image(pre_data)

            slice_index = random.randint(0, art_data.shape[2] - 1)

            x[i, :, :, 0] = art_data[:, :, slice_index]  # Channel 1
            x[i, :, :, 1] = ven_data[:, :, slice_index]  # Channel 2
            x[i, :, :, 2] = pre_data[:, :, slice_index]  # Channel 3
            y[i, :, :, 0] = mask_data[:, :, slice_index]

        return x, y


segmentation_files = sorted(glob(os.path.join(segmentation_folder, 'A21_*.nii')))

# Define the U-Net model
def encoder(inputs, filters, pool_size):
    conv_pool = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    conv_pool = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(conv_pool)
    return conv_pool

def decoder(inputs, concat_input, filters, transpose_size):
    up = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2DTranspose(filters, transpose_size, strides=(2, 2), padding='same')(inputs), concat_input])
    up = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(up)
    return up

def build_unet_model(img_size=(512, 512, 3)):  
    inputs = tf.keras.Input(img_size)

    # Encoder
    conv_pool1 = encoder(inputs, 32, (2, 2))
    conv_pool2 = encoder(conv_pool1, 64, (2, 2))
    conv_pool3 = encoder(conv_pool2, 128, (2, 2))
    conv_pool4 = encoder(conv_pool3, 256, (2, 2))

    # Bottleneck
    bridge = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv_pool4)

    # Decoder
    up6 = decoder(bridge, conv_pool3, 256, (2, 2))
    up7 = decoder(up6, conv_pool2, 128, (2, 2))
    up8 = decoder(up7, conv_pool1, 64, (2, 2))
    up9 = decoder(up8, inputs, 32, (2, 2))
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(up9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model


def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# 3-Fold Cross-Validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)

batch_size = 19
image_size = (256, 256)
fold_idx = 1

for train_index, val_index in kf.split(segmentation_files):
    print(f'Training fold {fold_idx}...')
    
    train_files = [segmentation_files[i] for i in train_index]
    val_files = [segmentation_files[i] for i in val_index]

    train_generator = NiiDataGenerator(train_files, batch_size, image_size)
    val_generator = NiiDataGenerator(val_files, batch_size, image_size)
    
    model = build_unet_model()
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

    model.fit(train_generator, validation_data=val_generator, epochs=1000,callbacks=[early_stopping])
    model_save_path = f'C:/Users/ester/Downloads/OneDrive_2024-08-16/LIG-AZifan Shared Folder/GEA21_Segmentation_Corrections_For_Ali/Munet_model_fold_{fold_idx}.h5'
##model.save(model_save_path)
##
##    model_save_path = f'Munet_model_fold_{fold_idx}.h5'
    model.save(model_save_path)
    print(f'Model for fold {fold_idx} saved to {model_save_path}')
    
    fold_idx += 1

print("3-Fold Cross-Validation completed.")
