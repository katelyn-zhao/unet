"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch.utils.data as data
import os.path
import nibabel as nib
import numpy as np
import os
import torch
from torch.utils.data import Dataset


def default_loader(path, target_z_size=40):
    # Load the NIfTI image
    img = nib.load(path).get_fdata()

    # Extract slices from 3D volume
    slices = []
    
    for i in range(img.shape[2]):
        slices.append(img[:, :, i])
    
    return slices

def default_flist_reader(flist):
    """
    flist format: impath\nimpath\n ...(each line contains the path to a .nii or .nii.gz file)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)
    return imlist



class ImageFilelist(Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader
        self.slices = []
        self.image_paths = []

        # Load slices for each image
        for impath in self.imlist:
            slices = self.loader(os.path.join(self.root, impath))
            self.slices.extend(slices)
            self.image_paths.extend([impath] * len(slices))

    def __getitem__(self, index):
        img = self.slices[index]
        img_path = self.image_paths[index]

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        if self.transform is not None:
            img = self.transform(img)

        return img, img_path

    def __len__(self):
        return len(self.slices)

class ImageLabelFilelist(Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(os.path.join(self.root, flist))
        self.transform = transform
        self.loader = loader

        self.classes = sorted(list(set([path.split('/')[0] for path in self.imlist])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

        self.imgs = []
        self.labels = []

        # Extract slices from each image and assign labels
        for impath in self.imlist:
            slices = self.loader(os.path.join(self.root, impath))
            for slice_img in slices:
                self.imgs.append(slice_img)
                self.labels.append(self.class_to_idx[impath.split('/')[0]])

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)

    

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.nii', '.nii.gz'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class ImageFolder(Dataset):
    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.loader = loader
        self.transform = transform
        self.return_paths = return_paths
        self.slices = []
        self.image_paths = []

        # Load slices for each image
        for path in imgs:
            slices = self.loader(path)
            self.slices.extend(slices)
            self.image_paths.extend([path] * len(slices))

    def __getitem__(self, index):
        img = self.slices[index]
        path = self.image_paths[index]

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        if self.transform is not None:
            img = self.transform(img)

        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.slices)
