# ============================================================
# visualization/visualizer.py
# Visualization System
# ============================================================

import cv2
import numpy as np

RISK_COLORS = {
    "LOW":      (0, 255, 0),
    "MEDIUM":   (0, 255, 255),
    "HIGH":     (0, 165, 255),
    "CRITICAL": (0, 0, 255),
}


def overlay_mask(image, mask):
    overlay = image.copy()
    overlay[mask > 0] = (255, 0, 0)
    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    return result


def draw_detections(image, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        risk     = det.get("risk", "LOW")
        track_id = det.get("track_id", -1)
        color    = RISK_COLORS.get(risk, (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"ID:{track_id} | {risk}"
        cv2.putText(image, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image


def create_heatmap_overlay(image, cam):
    orig_h, orig_w = image.shape[:2]
    cam_resized = cv2.resize(cam, (orig_w, orig_h))
    heatmap     = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    result      = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
    return result
