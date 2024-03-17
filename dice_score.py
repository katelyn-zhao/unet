import os
import io
import random
import nibabel
import numpy as np
import nibabel as nib
from nibabel import load
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import Sequence
from IPython.display import Image, display
from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import normalize
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from simple_unet_model import simple_unet_model
from skimage.transform import resize

def load_information(file_path):
    loss = []
    dice_scores = []
    val_loss = []
    val_dice_scores = []
    with open(file_path, 'r') as f:
        for line in f:
            if(line[0:5] == "Epoch"):
                continue
            else:
                info = line.split('-')
                loss.append(float(info[2][(info[2].index(':') + 1):]))
                dice_scores.append(float(info[3][(info[3].index(':') + 1):]))
                val_loss.append(float(info[4][(info[4].index(':') + 1):]))
                val_dice_scores.append(float(info[5][(info[5].index(':') + 1):]))
    return loss, dice_scores, val_loss, val_dice_scores

save_path = 'dice_scores.txt'
loss, dice_scores, val_loss, val_dice_scores = load_information(save_path)
epochs = range(1, len(dice_scores) + 1)

max_dice_score = max(dice_scores)
print(max_dice_score)

plt.figure(figsize=(12,3))
plt.subplot(1,2,1)
plt.title('BCE Losses v Epoch')
plt.plot(loss, color='r')
plt.plot(val_loss)
plt.ylabel('BCE Losses')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val.'], loc='upper right')
plt.subplot(1,2,2)
plt.title('Dice Score v Epoch')
plt.plot(dice_scores, color='r')
plt.plot(val_dice_scores)
plt.ylabel('Dice Score')
plt.xlabel('Epoch')
plt.tight_layout()
plt.show()