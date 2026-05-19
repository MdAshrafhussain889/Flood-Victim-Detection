# ============================================================
# segmentation/model_loader.py
# Segmentation Model Loader
# ============================================================

import torch
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.architectures import AttentionUNet, MobileNetUNet
from configs.config import BEST_MODEL, SEGMENTATION_MODEL


def load_segmentation_model():
    if SEGMENTATION_MODEL == "attention_unet":
        model = AttentionUNet()
    elif SEGMENTATION_MODEL == "mobilenet_unet":
        model = MobileNetUNet(pretrained=False)
    else:
        raise ValueError(f"Unknown model: {SEGMENTATION_MODEL}")

    if os.path.exists(BEST_MODEL):
        checkpoint = torch.load(BEST_MODEL, map_location="cpu")
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
        print(f"[Segmentation] Loaded weights from {BEST_MODEL}")
    else:
        print(f"[Segmentation] No weights found at {BEST_MODEL} - using random init")

    model.eval()
    return model
