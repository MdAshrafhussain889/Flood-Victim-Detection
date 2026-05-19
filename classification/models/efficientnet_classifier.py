# ============================================================
# classification/models/efficientnet_classifier.py
# EfficientNet-B0 Flood Classifier
# ============================================================

import torch
import torch.nn as nn
import torchvision.models as models


class FloodClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)
