# ============================================================
# detection/yolo_detector.py
# YOLOv8 Person Detection Engine
# ============================================================

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import YOLO_MODEL, YOLO_CONFIDENCE, PERSON_CLASS_ID


class YOLOPersonDetector:
    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO(YOLO_MODEL)

    def detect(self, image):
        results    = self.model(image, conf=YOLO_CONFIDENCE, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls != PERSON_CLASS_ID:
                    continue
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = (
                    box.xyxy[0].cpu().numpy().astype(int)
                )
                detections.append({
                    "box":        (x1, y1, x2, y2),
                    "confidence": confidence,
                })
        return detections
