# ============================================================
# segmentation/advanced_segmenter.py
# Advanced Flood Segmentation Engine - Core of v3
# ============================================================

import cv2
import torch
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from segmentation.model_loader import load_segmentation_model
from segmentation.preprocessing import preprocess_image
from segmentation.confidence import (
    compute_segmentation_confidence,
    should_suppress_mask,
)
from segmentation.postprocessing import remove_small_regions, apply_morphology
from configs.config import SEGMENTATION_THRESHOLD


class AdvancedFloodSegmenter:
    def __init__(self):
        self.model = load_segmentation_model()

    @torch.no_grad()
    def segment(self, image):
        original_h, original_w = image.shape[:2]
        tensor   = preprocess_image(image)
        logits   = self.model(tensor)
        prob_map = torch.sigmoid(logits)[0][0].cpu().numpy()

        confidence = compute_segmentation_confidence(prob_map)
        suppress   = should_suppress_mask(prob_map)

        if suppress:
            empty_mask = np.zeros((original_h, original_w), dtype=np.uint8)
            return {
                "mask":       empty_mask,
                "confidence": confidence,
                "suppressed": True,
            }

        mask = (prob_map > SEGMENTATION_THRESHOLD).astype(np.uint8) * 255
        mask = apply_morphology(mask)
        mask = remove_small_regions(mask)
        mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

        return {
            "mask":       mask,
            "confidence": confidence,
            "suppressed": False,
        }
