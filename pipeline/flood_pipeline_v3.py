# ============================================================
# pipeline/flood_pipeline_v3.py
# Integrated v3 Pipeline - Phase 2 + Phase 3
# ============================================================

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from classification.inference.classifier_engine import FloodClassifierEngine
from segmentation.advanced_segmenter import AdvancedFloodSegmenter


class FloodPipelineV3:
    def __init__(self):
        self.classifier = FloodClassifierEngine()
        self.segmenter  = AdvancedFloodSegmenter()

    def run(self, image):
        classification = self.classifier.predict(image)

        if not classification["run_segmentation"]:
            return {
                "decision":       "NO FLOOD",
                "segmentation":   None,
                "classification": classification,
            }

        segmentation = self.segmenter.segment(image)

        return {
            "decision":       "FLOOD",
            "classification": classification,
            "segmentation":   segmentation,
        }
