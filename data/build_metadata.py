# ============================================================
# data/build_metadata.py
# Unified Metadata Generator - Phase 1
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import (
    DATASET1_IMAGES,
    DATASET1_MASKS,
    DATASET2_FLOOD,
    DATASET2_NON_FLOOD,
    PROCESSED_IMAGES,
    PROCESSED_MASKS,
    METADATA_CSV,
    IMG_SIZE,
)

os.makedirs(PROCESSED_IMAGES, exist_ok=True)
os.makedirs(PROCESSED_MASKS,  exist_ok=True)

metadata = []


def process_dataset1():
    print("\nProcessing Dataset 1...")
    files = sorted(os.listdir(DATASET1_IMAGES))
    for filename in tqdm(files):
        img_path  = os.path.join(DATASET1_IMAGES, filename)
        mask_path = os.path.join(DATASET1_MASKS, filename)
        if not os.path.exists(mask_path):
            continue
        img  = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        img  = cv2.resize(img,  (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        save_img_name  = f"d1_{filename}"
        save_mask_name = f"d1_{filename}"
        save_img_path  = os.path.join(PROCESSED_IMAGES, save_img_name)
        save_mask_path = os.path.join(PROCESSED_MASKS,  save_mask_name)
        cv2.imwrite(save_img_path,  img)
        cv2.imwrite(save_mask_path, mask)
        metadata.append({
            "image_path": save_img_path,
            "mask_path":  save_mask_path,
            "label":      1,
            "source":     "dataset1",
        })


def process_dataset2_flood():
    print("\nProcessing Dataset 2 Flood Images...")
    files = sorted(os.listdir(DATASET2_FLOOD))
    for filename in tqdm(files):
        img_path = os.path.join(DATASET2_FLOOD, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        save_img_name  = f"d2_flood_{filename}"
        save_img_path  = os.path.join(PROCESSED_IMAGES, save_img_name)
        cv2.imwrite(save_img_path, img)
        empty_mask     = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        save_mask_name = f"d2_flood_{filename}"
        save_mask_path = os.path.join(PROCESSED_MASKS, save_mask_name)
        cv2.imwrite(save_mask_path, empty_mask)
        metadata.append({
            "image_path": save_img_path,
            "mask_path":  save_mask_path,
            "label":      1,
            "source":     "dataset2_flood",
        })


def process_dataset2_non_flood():
    print("\nProcessing Dataset 2 Non Flood Images...")
    files = sorted(os.listdir(DATASET2_NON_FLOOD))
    for filename in tqdm(files):
        img_path = os.path.join(DATASET2_NON_FLOOD, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        save_img_name  = f"d2_non_flood_{filename}"
        save_img_path  = os.path.join(PROCESSED_IMAGES, save_img_name)
        cv2.imwrite(save_img_path, img)
        empty_mask     = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        save_mask_name = f"d2_non_flood_{filename}"
        save_mask_path = os.path.join(PROCESSED_MASKS, save_mask_name)
        cv2.imwrite(save_mask_path, empty_mask)
        metadata.append({
            "image_path": save_img_path,
            "mask_path":  save_mask_path,
            "label":      0,
            "source":     "dataset2_non_flood",
        })


def save_metadata():
    os.makedirs(os.path.dirname(METADATA_CSV), exist_ok=True)
    df = pd.DataFrame(metadata)
    df.to_csv(METADATA_CSV, index=False)
    print("\nMetadata Saved")
    print(df.head())
    print(f"\nTotal Samples : {len(df)}")


if __name__ == "__main__":
    process_dataset1()
    process_dataset2_flood()
    process_dataset2_non_flood()
    save_metadata()
