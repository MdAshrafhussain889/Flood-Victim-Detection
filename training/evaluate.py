# ============================================================
# training/evaluate.py
# Test Set Evaluation
# ============================================================

import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import BEST_MODEL, OUTPUT_DIR, DEVICE
from models.architectures import get_model
from utils.losses_metrics import BCEDiceLoss, iou_score, dice_score, precision_score, recall_score, f1_score


@torch.no_grad()
def evaluate(model_name="attention_unet", test_loader=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = get_model(model_name)
    if os.path.exists(BEST_MODEL):
        ckpt = torch.load(BEST_MODEL, map_location="cpu")
        model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
        print(f"[Eval] Loaded weights from {BEST_MODEL}")
    model.eval()

    if test_loader is None:
        print("[Eval] No test loader provided. Please pass a test DataLoader.")
        return

    criterion = BCEDiceLoss()
    totals = {"loss": 0, "iou": 0, "dice": 0, "precision": 0, "recall": 0, "f1": 0}
    count  = 0
    samples = []

    for imgs, masks in tqdm(test_loader, desc="Evaluating", ncols=80):
        imgs   = imgs.to(DEVICE)
        masks  = masks.to(DEVICE)
        logits = model(imgs)
        loss   = criterion(logits, masks)
        totals["loss"]      += loss.item()
        totals["iou"]       += iou_score(logits, masks)
        totals["dice"]      += dice_score(logits, masks)
        totals["precision"] += precision_score(logits, masks)
        totals["recall"]    += recall_score(logits, masks)
        totals["f1"]        += f1_score(logits, masks)
        count += 1
        if len(samples) < 6:
            samples.append((imgs[0].cpu(), masks[0].cpu(), logits[0].cpu()))

    print("\n" + "=" * 50)
    print("  TEST SET EVALUATION REPORT")
    print("=" * 50)
    for k, v in totals.items():
        print(f"  {k.capitalize():<12} : {v / count:.4f}")
    print("=" * 50)

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(len(samples), 3, figsize=(9, 3 * len(samples)))
    for row, (img, mask, logit) in enumerate(samples):
        img_np  = (img.permute(1, 2, 0).numpy() * std + mean).clip(0, 1)
        gt_np   = mask.squeeze().numpy()
        pred_np = (torch.sigmoid(logit).squeeze() > 0.5).numpy()
        axes[row, 0].imshow(img_np);  axes[row, 0].set_title("Image");        axes[row, 0].axis("off")
        axes[row, 1].imshow(gt_np,  cmap="gray"); axes[row, 1].set_title("Ground Truth"); axes[row, 1].axis("off")
        axes[row, 2].imshow(pred_np, cmap="gray"); axes[row, 2].set_title("Prediction");   axes[row, 2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "test_predictions.png")
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"[Eval] Visual grid saved to {out_path}")


if __name__ == "__main__":
    evaluate()
