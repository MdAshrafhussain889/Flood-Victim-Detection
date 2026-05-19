# ============================================================
# utils/verify_dataset.py
# Dataset Verification System
# ============================================================

import os
import cv2
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import (
    DATASET1_IMAGES,
    DATASET1_MASKS,
    DATASET2_FLOOD,
    DATASET2_NON_FLOOD,
)

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def is_valid_image(path):
    img = cv2.imread(path)
    return img is not None


def verify_dataset1():
    print("\n" + "=" * 60)
    print("VERIFYING DATASET 1")
    print("=" * 60)
    image_files = sorted(os.listdir(DATASET1_IMAGES))
    mask_files  = sorted(os.listdir(DATASET1_MASKS))
    image_set   = set(image_files)
    mask_set    = set(mask_files)
    missing_masks  = image_set - mask_set
    missing_images = mask_set - image_set
    print(f"Total Images   : {len(image_files)}")
    print(f"Total Masks    : {len(mask_files)}")
    print(f"Missing Masks  : {len(missing_masks)}")
    print(f"Missing Images : {len(missing_images)}")
    valid_pairs = []
    for filename in image_set.intersection(mask_set):
        img_path  = os.path.join(DATASET1_IMAGES, filename)
        mask_path = os.path.join(DATASET1_MASKS, filename)
        if not is_valid_image(img_path):
            continue
        if not is_valid_image(mask_path):
            continue
        valid_pairs.append(filename)
    print(f"Valid Pairs    : {len(valid_pairs)}")
    return valid_pairs


def verify_dataset2():
    print("\n" + "=" * 60)
    print("VERIFYING DATASET 2")
    print("=" * 60)
    flood_images     = []
    non_flood_images = []
    for file in os.listdir(DATASET2_FLOOD):
        if file.lower().endswith(VALID_EXTENSIONS):
            path = os.path.join(DATASET2_FLOOD, file)
            if is_valid_image(path):
                flood_images.append(file)
    for file in os.listdir(DATASET2_NON_FLOOD):
        if file.lower().endswith(VALID_EXTENSIONS):
            path = os.path.join(DATASET2_NON_FLOOD, file)
            if is_valid_image(path):
                non_flood_images.append(file)
    print(f"Flood Images     : {len(flood_images)}")
    print(f"Non Flood Images : {len(non_flood_images)}")
    return flood_images, non_flood_images


if __name__ == "__main__":
    verify_dataset1()
    verify_dataset2()
