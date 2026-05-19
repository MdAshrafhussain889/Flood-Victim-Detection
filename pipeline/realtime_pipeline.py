# ============================================================
# pipeline/realtime_pipeline.py
# Real-Time Flood System - Central AI Engine of v3
# ============================================================

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pipeline.flood_pipeline_v3 import FloodPipelineV3
from detection.yolo_detector import YOLOPersonDetector
from tracking.simple_tracker import SimpleTracker
from risk_engine.adaptive_risk import (
    compute_overlap,
    compute_local_water_density,
    compute_body_visibility,
    compute_risk_score,
    classify_risk,
)


class RealTimeFloodSystem:
    def __init__(self):
        self.pipeline = FloodPipelineV3()
        self.detector = YOLOPersonDetector()
        self.tracker  = SimpleTracker()

    def process_frame(self, image):
        result = self.pipeline.run(image)

        if result["decision"] == "NO FLOOD":
            result["detections"] = []
            return result

        mask       = result["segmentation"]["mask"]
        confidence = result["segmentation"]["confidence"]

        detections = self.detector.detect(image)
        detections = self.tracker.update(detections)

        for det in detections:
            box        = det["box"]
            overlap    = compute_overlap(mask, box)
            density    = compute_local_water_density(mask, box)
            visibility = compute_body_visibility(box)
            score      = compute_risk_score(overlap, density, confidence, visibility)
            risk       = classify_risk(score)
            det["risk"]       = risk
            det["risk_score"] = score
            det["overlap"]    = round(overlap * 100, 1)

        result["detections"] = detections
        return result
