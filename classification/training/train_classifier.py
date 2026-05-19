# ============================================================
# classification/training/train_classifier.py
# Classifier Training Pipeline
# ============================================================

import os
import sys
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from classification.dataset import FloodClassificationDataset
from classification.transforms import get_train_transform, get_val_transform
from classification.models.efficientnet_classifier import FloodClassifier
from classification.metrics import compute_metrics
from configs.config import (
    TRAIN_CSV,
    VAL_CSV,
    CLASSIFIER_BATCH_SIZE,
    CLASSIFIER_EPOCHS,
    CLASSIFIER_LR,
    CLASSIFIER_WEIGHT_DECAY,
    CLASSIFIER_CHECKPOINT,
    CLASSIFIER_PATIENCE,
)

DEVICE = "cpu"

train_dataset = FloodClassificationDataset(TRAIN_CSV, transform=get_train_transform())
val_dataset   = FloodClassificationDataset(VAL_CSV,   transform=get_val_transform())

train_loader = DataLoader(train_dataset, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=CLASSIFIER_BATCH_SIZE, shuffle=False, num_workers=0)

model     = FloodClassifier(pretrained=True).to(DEVICE)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=CLASSIFIER_LR, weight_decay=CLASSIFIER_WEIGHT_DECAY)

best_val_f1      = 0
patience_counter = 0

for epoch in range(CLASSIFIER_EPOCHS):
    print("\n" + "=" * 60)
    print(f"Epoch {epoch + 1}/{CLASSIFIER_EPOCHS}")
    print("=" * 60)

    model.train()
    train_logits  = []
    train_targets = []
    running_loss  = 0
    train_bar     = tqdm(train_loader)

    for batch in train_bar:
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_logits.append(logits.detach())
        train_targets.append(labels.detach())
        train_bar.set_description(f"Train Loss: {loss.item():.4f}")

    train_logits  = torch.cat(train_logits)
    train_targets = torch.cat(train_targets)
    train_metrics = compute_metrics(train_logits, train_targets)

    model.eval()
    val_logits  = []
    val_targets = []
    val_loss    = 0

    with torch.no_grad():
        val_bar = tqdm(val_loader)
        for batch in val_bar:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits = model(images)
            loss   = criterion(logits, labels)
            val_loss += loss.item()
            val_logits.append(logits)
            val_targets.append(labels)
            val_bar.set_description(f"Val Loss: {loss.item():.4f}")

    val_logits  = torch.cat(val_logits)
    val_targets = torch.cat(val_targets)
    val_metrics = compute_metrics(val_logits, val_targets)

    print("\nTrain Metrics:", train_metrics)
    print("Validation Metrics:", val_metrics)

    if val_metrics["f1"] > best_val_f1:
        best_val_f1 = val_metrics["f1"]
        patience_counter = 0
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), CLASSIFIER_CHECKPOINT)
        print("\nBest classifier saved")
    else:
        patience_counter += 1
        if patience_counter >= CLASSIFIER_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
