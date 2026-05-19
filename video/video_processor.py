# ============================================================
# video/video_processor.py
# Video Processing Module
# ============================================================

import cv2
import sys, os
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pipeline.realtime_pipeline import RealTimeFloodSystem
from visualization.visualizer import overlay_mask, draw_detections
from configs.config import VIDEO_SKIP_FRAMES, VIDEO_OUTPUT_FPS


class VideoProcessor:
    def __init__(self):
        self.system = RealTimeFloodSystem()

    def process(self, input_path, output_path):
        cap    = cv2.VideoCapture(input_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            VIDEO_OUTPUT_FPS,
            (width, height),
        )

        frame_index = 0
        progress    = tqdm(total=total_frames, desc="Processing video")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % VIDEO_SKIP_FRAMES == 0:
                result = self.system.process_frame(frame)
                if result["decision"] == "FLOOD":
                    mask  = result["segmentation"]["mask"]
                    frame = overlay_mask(frame, mask)
                    frame = draw_detections(frame, result["detections"])

            writer.write(frame)
            frame_index += 1
            progress.update(1)

        cap.release()
        writer.release()
        progress.close()
        print(f"\nVideo saved to: {output_path}")
        return output_path
