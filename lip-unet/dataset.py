""" 
dataset.py

This file is responsible for:
> creating image and mask list while mapping that both should match each other
> checking that both image and mask count should match
> applying transformation on both image and mask so that no mismatch occurs
> customizing the dataset where we load both image and mask, index it, read it and convert into tensor
> splitting the image and mask files into train-80%, valid-10%, test-10%
> packing into DataLoader with batch_size
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


def file_map(masks_path, images_path):
    images_filename = []
    mask_filename = []
    for filename in os.listdir(images_path):
        image_path = os.path.join(images_path, filename)
        mask_name = filename.replace("image", "mask").replace(".jpg", ".png")
        mask_paths = os.path.join(masks_path, mask_name)

     
        # only append pairs where the mask file actually exists
        if os.path.exists(mask_paths):
            images_filename.append(image_path)
            mask_filename.append(mask_paths)
        else:
            print("skipping, mask missing:", mask_paths)

    return images_filename, mask_filename


def check_idx(image_filenames, mask_filenames):
    missed_file = []

    if len(image_filenames) != len(mask_filenames):
        print("Length mismatch:", len(image_filenames), len(mask_filenames))

    for idx in range(len(image_filenames)):
        image_id = os.path.basename(image_filenames[idx]).replace("image", "").replace(".jpg", "")
        mask_id = os.path.basename(mask_filenames[idx]).replace("mask", "").replace(".png", "")

        # if the image and mask filename do not match, record it
        if image_id != mask_id:
            missed_file.append((idx, image_filenames[idx], mask_filenames[idx]))

    return missed_file


masks_path = "path to the mask_path"
images_path = "path to the image_path"
images_filename, mask_filename = file_map(masks_path, images_path)
mismatches = check_idx(images_filename, mask_filename)

if mismatches:
    print(f"Found {len(mismatches)} mismatches:")
    print(mismatches[:5])
else:
    print("No mismatches found, all image/mask pairs align.")


train_transforms = A.Compose([
    A.Resize(height=256, width=256),
    A.Rotate(limit=35, p=0.5),
    A.HorizontalFlip(p=0.5),  # 50% chance of flipping horizontally
    # normalize so that pixel values are in 0-1 range
    A.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        max_pixel_value=255.0
    ),
    ToTensorV2(),  # convert augmented numpy arrays into tensors
])

valid_transforms = A.Compose([
    A.Resize(height=256, width=256),
    A.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        max_pixel_value=255.0
    ),
    ToTensorV2()
])


class Custom(Dataset):
    def __init__(self, image_path, mask_path, transforms=None):
        self.transforms = transforms
        self.image_path = image_path
        self.mask_path = mask_path

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image_file = self.image_path[idx]
        mask_file = self.mask_path[idx]

        # load image from disk and convert BGR (OpenCV default) → RGB
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # load mask as grayscale (single channel)
        mask = cv2.imread(mask_file, 0)
        # binarize mask: any pixel > 0 becomes 1 (foreground), rest stays 0 (background)
        mask = np.where(mask > 0, 1, 0).astype(np.uint8)

        # apply transformation to both image and mask together to keep them aligned
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


# wrapper that applies its own transform on top of a random_split Subset
class TransformedSubset(Dataset):
    def __init__(self, subset, transforms):
        self.subset = subset
        self.transforms = transforms

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # base dataset has transforms=None so we get raw numpy arrays here
        image, mask = self.subset[idx]
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        # add channel dim to mask: [H, W] → [1, H, W] and cast to float for loss functions
        mask = mask.unsqueeze(0).float()
        return image, mask


full_dataset = Custom(images_filename, mask_filename, transforms=None)  # no transform at base level

train_ratio = int(len(full_dataset) * 0.8)
valid_ratio = int(len(full_dataset) * 0.1)
test_ratio = len(full_dataset) - train_ratio - valid_ratio  # remaining to avoid rounding issues

# fixed seed so train/valid/test split is always the same across runs
generator = torch.Generator().manual_seed(42)
train_subset, valid_subset, test_subset = random_split(
    full_dataset, [train_ratio, valid_ratio, test_ratio], generator=generator
)

# apply the right transforms to each split
train_dataset = TransformedSubset(train_subset, train_transforms)
valid_dataset = TransformedSubset(valid_subset, valid_transforms)
test_dataset = TransformedSubset(test_subset, valid_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=8, num_workers=2, pin_memory=True)