# ============================================================
# segmentation/postprocessing.py
# Morphological Cleanup System
# ============================================================

import cv2
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import MIN_REGION_AREA, MORPH_KERNEL_SIZE


def remove_small_regions(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= MIN_REGION_AREA:
            cleaned[labels == i] = 255
    return cleaned


def apply_morphology(mask):
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return mask
