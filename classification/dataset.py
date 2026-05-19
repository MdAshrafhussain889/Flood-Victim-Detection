# ============================================================
# classification/dataset.py
# Classification Dataset Loader
# ============================================================

import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class FloodClassificationDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = cv2.imread(row.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(row.label)
        if self.transform:
            augmented = self.transform(image=image)
            image     = augmented["image"]
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.float32),
        }
