# Flood Victim Detection v3

A real-time disaster-response AI platform built with PyTorch, YOLOv8, and Streamlit.

## Phase Overview

| Phase | Description |
|-------|-------------|
| Phase 1 | Unified Dataset Pipeline |
| Phase 2 | Flood/No-Flood Classification (EfficientNet-B0) |
| Phase 3 | Advanced Segmentation + Adaptive Risk Engine |
| Phase 4 | YOLOv8 Detection + Tracking + Streamlit v3 Dashboard |

## Pipeline

```
Input Image / Video / Webcam
        |
Flood Classifier (EfficientNet-B0)
        |
    NO FLOOD --> Stop
        |
   FLOOD
        |
Segmentation (Attention U-Net)
        |
Confidence Filtering + Morphological Cleanup
        |
YOLOv8 Person Detection
        |
Centroid Tracking
        |
Adaptive Risk Engine
        |
Visualization Layer
        |
Streamlit Dashboard
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Phase 1 - Build Dataset
```bash
python utils/verify_dataset.py
python data/build_metadata.py
python data/create_splits.py
```

### Phase 2 - Train Classifier
```bash
python classification/training/train_classifier.py
```

### Phase 3 - Train Segmentation Model
```bash
python training/train.py
python training/train.py --resume
```

### Phase 4 - Launch Dashboard
```bash
streamlit run streamlit_app/app.py
```

## CPU Performance Expectations

| Task | Speed |
|------|-------|
| Classification | 20-40 ms |
| Segmentation | 120-250 ms |
| YOLO Detection | 60-120 ms |
| Full Pipeline | 300-600 ms |
| Video FPS | 2-5 FPS CPU |

## Project Structure

```
flood_victim_detection_v3/
|-- configs/             Config file
|-- models/              Model architectures
|-- data/                Dataset pipeline
|-- utils/               Utilities and metrics
|-- classification/      Phase 2 - Classifier
|-- segmentation/        Phase 3 - Segmentation
|-- risk_engine/         Adaptive risk scoring
|-- detection/           YOLOv8 detector
|-- tracking/            Centroid tracker
|-- explainability/      Grad-CAM
|-- visualization/       Drawing utilities
|-- pipeline/            Integrated pipelines
|-- video/               Video processor
|-- training/            Training scripts
|-- streamlit_app/       Dashboard UI
|-- checkpoints/         Saved models
|-- outputs/             Results
```

## Risk Formula

```
Risk Score =
    0.40 x overlap
  + 0.30 x water_density
  + 0.20 x segmentation_confidence
  + 0.10 x body_visibility
```

| Score | Risk Level |
|-------|-----------|
| >= 0.75 | CRITICAL |
| >= 0.50 | HIGH |
| >= 0.25 | MEDIUM |
| < 0.25  | LOW |
