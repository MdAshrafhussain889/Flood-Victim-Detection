# ============================================================
# classification/inference/classifier_engine.py
# Flood Classifier Inference Engine - Stage 1 of v3 Pipeline
# ============================================================

import cv2
import torch
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from classification.models.efficientnet_classifier import FloodClassifier
from configs.config import (
    CLASSIFIER_CHECKPOINT,
    CLASSIFIER_THRESHOLD,
    CLASSIFIER_IMG_SIZE,
)

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


class FloodClassifierEngine:
    def __init__(self):
        self.model = FloodClassifier(pretrained=False)
        self.model.load_state_dict(
            torch.load(CLASSIFIER_CHECKPOINT, map_location="cpu")
        )
        self.model.eval()

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (CLASSIFIER_IMG_SIZE, CLASSIFIER_IMG_SIZE))
        image = image.astype(np.float32) / 255.0
        image = (image - MEAN) / STD
        image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
        return image.float()

    @torch.no_grad()
    def predict(self, image):
        tensor      = self.preprocess(image)
        logits      = self.model(tensor)
        probability = torch.sigmoid(logits).item()
        is_flood    = probability > CLASSIFIER_THRESHOLD
        return {
            "flood_probability": probability,
            "decision":          "FLOOD" if is_flood else "NO FLOOD",
            "run_segmentation":  is_flood,
        }
