# ============================================================
# risk_engine/adaptive_risk.py
# Adaptive Risk Scoring Engine
#
# Risk Score =
#   0.40 x overlap
#   0.30 x water_density
#   0.20 x segmentation_confidence
#   0.10 x body_visibility
# ============================================================

import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import (
    RISK_LOW_THRESHOLD,
    RISK_MEDIUM_THRESHOLD,
    RISK_HIGH_THRESHOLD,
)


def compute_body_visibility(box):
    x1, y1, x2, y2 = box
    height = y2 - y1
    width  = x2 - x1
    area   = width * height
    return min(area / 50000.0, 1.0)


def compute_local_water_density(mask, box):
    x1, y1, x2, y2 = box
    region = mask[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0
    return float(np.mean(region > 0))


def compute_overlap(mask, box):
    x1, y1, x2, y2 = box
    region = mask[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0
    return float(np.mean(region > 0))


def compute_risk_score(overlap, density, confidence, visibility):
    score = (
        0.40 * overlap
        + 0.30 * density
        + 0.20 * confidence
        + 0.10 * visibility
    )
    return score


def classify_risk(score):
    if score >= RISK_HIGH_THRESHOLD:
        return "CRITICAL"
    elif score >= RISK_MEDIUM_THRESHOLD:
        return "HIGH"
    elif score >= RISK_LOW_THRESHOLD:
        return "MEDIUM"
    else:
        return "LOW"
