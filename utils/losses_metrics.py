# ============================================================
# utils/losses_metrics.py
# Loss Functions and Metrics
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


# ============================================================
# Loss Functions
# ============================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs   = torch.sigmoid(logits)
        probs   = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        return (self.bce_weight * self.bce(logits, targets) +
                self.dice_weight * self.dice(logits, targets))


# ============================================================
# Metric Functions
# ============================================================

def _binarise(logits, threshold=0.5):
    return (torch.sigmoid(logits) > threshold).float()


def iou_score(logits, targets, smooth=1e-6):
    preds   = _binarise(logits).view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    union        = preds.sum() + targets.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


def dice_score(logits, targets, smooth=1e-6):
    preds   = _binarise(logits).view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return ((2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)).item()


def precision_score(logits, targets, smooth=1e-6):
    preds   = _binarise(logits).view(-1)
    targets = targets.view(-1)
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    return ((tp + smooth) / (tp + fp + smooth)).item()


def recall_score(logits, targets, smooth=1e-6):
    preds   = _binarise(logits).view(-1)
    targets = targets.view(-1)
    tp = (preds * targets).sum()
    fn = ((1 - preds) * targets).sum()
    return ((tp + smooth) / (tp + fn + smooth)).item()


def f1_score(logits, targets, smooth=1e-6):
    p = precision_score(logits, targets, smooth)
    r = recall_score(logits, targets, smooth)
    return (2.0 * p * r / (p + r + smooth))


# ============================================================
# Metric Accumulator
# ============================================================

class MetricTracker:
    def __init__(self):
        self._sums   = defaultdict(float)
        self._counts = defaultdict(int)

    def update(self, loss, logits, targets):
        n = targets.shape[0]
        self._sums["loss"]      += loss * n
        self._sums["iou"]       += iou_score(logits, targets) * n
        self._sums["dice"]      += dice_score(logits, targets) * n
        self._sums["precision"] += precision_score(logits, targets) * n
        self._sums["recall"]    += recall_score(logits, targets) * n
        self._sums["f1"]        += f1_score(logits, targets) * n
        self._counts["total"]   += n

    def compute(self):
        n = self._counts["total"]
        if n == 0:
            return {}
        return {k: v / n for k, v in self._sums.items()}

    def reset(self):
        self._sums.clear()
        self._counts.clear()

    def pretty(self, prefix=""):
        d = self.compute()
        parts = [
            f"Loss={d.get('loss', 0):.4f}",
            f"IoU={d.get('iou', 0):.4f}",
            f"Dice={d.get('dice', 0):.4f}",
            f"F1={d.get('f1', 0):.4f}",
        ]
        line = " | ".join(parts)
        return f"{prefix}  {line}" if prefix else line
