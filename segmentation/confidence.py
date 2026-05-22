# ============================================================
# segmentation/confidence.py
# Confidence Suppression System
# ============================================================

import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import MIN_FLOOD_CONFIDENCE, SEGMENTATION_THRESHOLD


def compute_segmentation_confidence(prob_map):
    flood_pixels = prob_map >= SEGMENTATION_THRESHOLD
    return float(np.count_nonzero(flood_pixels) / flood_pixels.size)


def should_suppress_mask(prob_map):
    confidence = compute_segmentation_confidence(prob_map)
    return confidence < MIN_FLOOD_CONFIDENCE
