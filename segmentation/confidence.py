# ============================================================
# segmentation/confidence.py
# Confidence Suppression System
# ============================================================

import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import MIN_FLOOD_CONFIDENCE


def compute_segmentation_confidence(prob_map):
    return float(np.mean(prob_map))


def should_suppress_mask(prob_map):
    confidence = compute_segmentation_confidence(prob_map)
    return confidence < MIN_FLOOD_CONFIDENCE
