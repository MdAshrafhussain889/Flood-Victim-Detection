# ============================================================
# training/train.py
# CPU Optimized Training System
# ============================================================

import os
import sys
import time
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import (
    EPOCHS, LR, WEIGHT_DECAY, PATIENCE, ACCUM_STEPS,
    MODEL_NAME, BEST_MODEL, CHECKPOINT_DIR, OUTPUT_DIR,
    BCE_WEIGHT, DICE_WEIGHT, SEED, DEVICE,
    IMAGE_DIR, MASK_DIR, BATCH_SIZE, NUM_WORKERS, IMG_SIZE,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
)
from models.architectures import get_model
from utils.losses_metrics import BCEDiceLoss, MetricTracker

import cv2
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def verify_dataset(img_dir=IMAGE_DIR, mask_dir=MASK_DIR):
    img_files  = set(os.listdir(img_dir))
    mask_files = set(os.listdir(mask_dir))
    paired = img_files & mask_files
    valid  = []
    for fname in sorted(paired):
        img  = cv2.imread(os.path.join(img_dir, fname))
        mask = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is not None and mask is not None:
            valid.append(fname)
    print(f"[Dataset] Valid pairs: {len(valid)}")
    return valid


def split_dataset(filenames, seed=SEED):
    train, temp = train_test_split(filenames, test_size=(1.0 - TRAIN_RATIO), random_state=seed)
    val_frac    = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val, test   = train_test_split(temp, test_size=(1.0 - val_frac), random_state=seed)
    print(f"[Split] Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test


def get_train_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussNoise(std_range=(0.04, 0.2), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class FloodDataset(Dataset):
    def __init__(self, filenames, img_dir=IMAGE_DIR, mask_dir=MASK_DIR, transform=None):
        self.filenames = filenames
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img   = cv2.imread(os.path.join(self.img_dir, fname))
        img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(os.path.join(self.mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        mask  = (mask > 127).astype(np.float32)
        if self.transform:
            aug  = self.transform(image=img, mask=mask)
            img  = aug["image"]
            mask = aug["mask"]
        return img, mask.unsqueeze(0)


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    if not os.path.exists(path):
        print(f"[Resume] No checkpoint at {path} - starting fresh.")
        return 0
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    epoch = ckpt.get("epoch", 0)
    print(f"[Resume] Loaded checkpoint from epoch {epoch}")
    return epoch


def plot_curves(history, save_dir=OUTPUT_DIR):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(epochs, history["train_iou"], label="Train IoU")
    axes[1].plot(epochs, history["val_iou"],   label="Val IoU")
    axes[1].set_title("IoU Curves")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"[Plot] Saved to {path}")


def train_one_epoch(model, loader, criterion, optimizer, accum_steps=ACCUM_STEPS):
    model.train()
    tracker = MetricTracker()
    optimizer.zero_grad()
    pbar = tqdm(enumerate(loader), total=len(loader), desc="  Train", leave=False, ncols=90)
    for step, (imgs, masks) in pbar:
        imgs  = imgs.to(DEVICE)
        masks = masks.to(DEVICE)
        logits = model(imgs)
        loss   = criterion(logits, masks)
        (loss / accum_steps).backward()
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()
        tracker.update(loss.item(), logits.detach(), masks)
        pbar.set_postfix_str(tracker.pretty())
    return tracker.compute()


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    tracker = MetricTracker()
    pbar = tqdm(loader, total=len(loader), desc="  Val  ", leave=False, ncols=90)
    for imgs, masks in pbar:
        imgs  = imgs.to(DEVICE)
        masks = masks.to(DEVICE)
        logits = model(imgs)
        loss   = criterion(logits, masks)
        tracker.update(loss.item(), logits, masks)
        pbar.set_postfix_str(tracker.pretty())
    return tracker.compute()


def train(resume=False):
    set_seed()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    valid_files = verify_dataset()
    train_f, val_f, test_f = split_dataset(valid_files)

    train_ds = FloodDataset(train_f, transform=get_train_transform())
    val_ds   = FloodDataset(val_f,   transform=get_val_transform())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = get_model(MODEL_NAME)

    unfreeze_at = None
    if MODEL_NAME == "mobilenet_unet":
        model.freeze_encoder()
        unfreeze_at = 5

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = BCEDiceLoss(bce_weight=BCE_WEIGHT, dice_weight=DICE_WEIGHT)

    resume_path = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")
    start_epoch = 0
    if resume:
        start_epoch = load_checkpoint(resume_path, model, optimizer, scheduler)

    history = {
        "train_loss": [], "val_loss": [],
        "train_iou":  [], "val_iou":  [],
        "train_dice": [], "val_dice": [],
    }

    best_val_loss    = float("inf")
    patience_counter = 0

    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        print(f"\n{'--' * 30}")
        print(f"Epoch {epoch+1}/{EPOCHS}  |  LR: {optimizer.param_groups[0]['lr']:.2e}")

        if unfreeze_at and epoch == unfreeze_at:
            model.unfreeze_encoder()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR * 0.1, weight_decay=WEIGHT_DECAY)

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer)
        val_metrics   = validate(model, val_loader, criterion)
        elapsed       = time.time() - epoch_start

        print(
            f"  Train | Loss={train_metrics['loss']:.4f} "
            f"IoU={train_metrics['iou']:.4f} Dice={train_metrics['dice']:.4f}"
        )
        print(
            f"  Val   | Loss={val_metrics['loss']:.4f} "
            f"IoU={val_metrics['iou']:.4f} Dice={val_metrics['dice']:.4f} "
            f"({elapsed:.1f}s)"
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_iou"].append(train_metrics["iou"])
        history["val_iou"].append(val_metrics["iou"])
        history["train_dice"].append(train_metrics["dice"])
        history["val_dice"].append(val_metrics["dice"])

        scheduler.step(val_metrics["loss"])

        save_checkpoint({
            "epoch":           epoch + 1,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_loss":        val_metrics["loss"],
        }, resume_path)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss    = val_metrics["loss"]
            patience_counter = 0
            save_checkpoint({"model_state": model.state_dict()}, BEST_MODEL)
            print(f"  Best model saved  (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print(f"\n[EarlyStopping] Triggered at epoch {epoch+1}.")
            break

    plot_curves(history)
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {BEST_MODEL}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    train(resume=args.resume)
