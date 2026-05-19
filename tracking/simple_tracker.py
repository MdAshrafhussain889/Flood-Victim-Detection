# ============================================================
# tracking/simple_tracker.py
# Centroid-Based Victim Tracking System
# ============================================================

import math
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import TRACK_MAX_DISTANCE


class SimpleTracker:
    def __init__(self):
        self.next_id = 0
        self.tracks  = {}

    def centroid(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def update(self, detections):
        updated_tracks = {}
        for detection in detections:
            box    = detection["box"]
            center = self.centroid(box)
            matched_id   = None
            min_distance = float("inf")
            for track_id, prev_center in self.tracks.items():
                dist = self.distance(center, prev_center)
                if dist < TRACK_MAX_DISTANCE and dist < min_distance:
                    min_distance = dist
                    matched_id   = track_id
            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1
            updated_tracks[matched_id] = center
            detection["track_id"] = matched_id
        self.tracks = updated_tracks
        return detections
