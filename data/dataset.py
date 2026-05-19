# ============================================================
# data/dataset.py
# Unified Dataset Loader - supports classification + segmentation
# ============================================================

import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import (
    IMG_SIZE, SEED, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
)

IMAGE_DIR = None
MASK_DIR  = None
try:
    from configs.config import IMAGE_DIR, MASK_DIR
except ImportError:
    pass


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[Seed] Set to {seed}")


def get_train_transform(img_size=IMG_SIZE):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussNoise(std_range=(0.04, 0.2), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform(img_size=IMG_SIZE):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class FloodDataset(Dataset):
    def __init__(self, filenames, img_dir, mask_dir, transform=None):
        self.filenames = filenames
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img   = cv2.imread(os.path.join(self.img_dir, fname))
        img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(os.path.join(self.mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        mask  = (mask > 127).astype(np.float32)
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img  = augmented["image"]
            mask = augmented["mask"]
        mask = mask.unsqueeze(0)
        return img, mask
