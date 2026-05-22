# ============================================================
# classification/inference/classifier_engine.py
# Flood Classifier Inference Engine - Stage 1 of v3 Pipeline
# ============================================================

import cv2
import torch
import numpy as np
import sys, os
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from classification.models.efficientnet_classifier import FloodClassifier
from configs.config import (
    CLASSIFIER_CHECKPOINT,
    CLASSIFIER_THRESHOLD,
    CLASSIFIER_IMG_SIZE,
    DEVICE,
)

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


class FloodClassifierEngine:
    def __init__(self):
        if not os.path.exists(CLASSIFIER_CHECKPOINT):
            raise FileNotFoundError(
                "Flood classifier checkpoint not found: "
                f"{CLASSIFIER_CHECKPOINT}. Set CLASSIFIER_CHECKPOINT to a trained model."
            )
        self.model = FloodClassifier(pretrained=False)
        state_dict = self._load_checkpoint(CLASSIFIER_CHECKPOINT)
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Flood classifier checkpoint could not be loaded into the "
                f"model architecture: {CLASSIFIER_CHECKPOINT}"
            ) from exc
        self.model.to(DEVICE).eval()

    @staticmethod
    def _load_checkpoint(checkpoint_path):
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=DEVICE,
                weights_only=True,
            )
        except TypeError:
            warnings.warn(
                "This PyTorch version does not support torch.load(weights_only=True); "
                "loading the classifier checkpoint with the legacy loader.",
                RuntimeWarning,
            )
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break

        if not isinstance(checkpoint, dict) or not checkpoint:
            raise RuntimeError(
                "Flood classifier checkpoint is empty or is not a PyTorch "
                f"state_dict: {checkpoint_path}"
            )

        return checkpoint

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (CLASSIFIER_IMG_SIZE, CLASSIFIER_IMG_SIZE))
        image = image.astype(np.float32) / 255.0
        image = (image - MEAN) / STD
        image = np.ascontiguousarray(image)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return tensor.float().to(DEVICE)

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
