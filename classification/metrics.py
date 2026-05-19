# ============================================================
# classification/metrics.py
# Classification Metrics
# ============================================================

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def compute_metrics(logits, targets):
    probs   = torch.sigmoid(logits)
    preds   = (probs > 0.5).int().cpu().numpy()
    targets = targets.int().cpu().numpy()
    probs   = probs.detach().cpu().numpy()
    metrics = {
        "accuracy":  accuracy_score(targets, preds),
        "precision": precision_score(targets, preds, zero_division=0),
        "recall":    recall_score(targets, preds, zero_division=0),
        "f1":        f1_score(targets, preds, zero_division=0),
        "auc":       roc_auc_score(targets, probs) if len(set(targets)) > 1 else 0.0,
    }
    return metrics
