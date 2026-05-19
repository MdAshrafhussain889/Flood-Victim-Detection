# ============================================================
# classification/transforms.py
# Augmentation Pipeline for Classification
# ============================================================

import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import CLASSIFIER_IMG_SIZE


def get_train_transform():
    return A.Compose([
        A.Resize(CLASSIFIER_IMG_SIZE, CLASSIFIER_IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.GaussNoise(std_range=(0.02, 0.10), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform():
    return A.Compose([
        A.Resize(CLASSIFIER_IMG_SIZE, CLASSIFIER_IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
