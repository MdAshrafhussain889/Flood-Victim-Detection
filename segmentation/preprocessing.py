# ============================================================
# segmentation/preprocessing.py
# Segmentation Preprocessing System
# ============================================================

import cv2
import torch
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import IMG_SIZE

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    tensor = torch.tensor(image)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor.float()
