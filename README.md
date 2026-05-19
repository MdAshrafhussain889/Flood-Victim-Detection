# Flood Victim Detection and Risk Analysis System 

AI-powered real-time flood monitoring, victim detection, segmentation, and adaptive risk analysis system using Deep Learning and Computer Vision.

---

## Project Overview

Flood Victim Detection v3 is an advanced AI surveillance and disaster-management platform designed to:

- Detect flooded regions
- Identify trapped victims
- Analyze flood severity
- Estimate human risk levels
- Provide real-time monitoring dashboard

The system combines:

- Attention U-Net Flood Segmentation
- YOLOv8 Human Detection
- Adaptive Risk Engine
- Victim Tracking
- Real-Time AI Dashboard

---

# Key Features

- Flood Classification
- Floodwater Segmentation
- Victim Detection
- Real-Time Risk Analysis
- Human Tracking
- Webcam Monitoring
- Video Processing
- Flood Overlay Visualization
- AI Analytics Dashboard
- Streamlit-Based UI

---

# System Architecture

```text
Input Image / Video / Webcam
            │
            ▼
 Flood Classification Model
            │
     ┌──────┴──────┐
     │             │
 NO FLOOD      FLOOD DETECTED
                     │
                     ▼
        Attention U-Net Segmentation
                     │
                     ▼
          YOLOv8 Victim Detection
                     │
                     ▼
            Person Tracking Module
                     │
                     ▼
            Adaptive Risk Engine
                     │
                     ▼
        Real-Time AI Dashboard UI    



````

---

# AI Models Used

| Component            | Model                |
| -------------------- | -------------------- |
| Flood Classification | CNN Classifier       |
| Flood Segmentation   | Attention U-Net      |
| Victim Detection     | YOLOv8n              |
| Tracking             | Centroid Tracker     |
| Risk Analysis        | Adaptive Risk Engine |

---

# Dataset Information

## Flood Segmentation Dataset

* Total Images: 3401
* Binary Flood Masks
* Train / Validation / Test Split

| Split      | Count |
| ---------- | ----- |
| Train      | 2380  |
| Validation | 510   |
| Test       | 511   |

---

# Achieved Results

| Metric                   | Score     |
| ------------------------ | --------- |
| Dice Score               | 0.86      |
| IoU Score                | 0.76      |
| Validation Loss          | 0.1515    |
| Real-Time Inference      | Supported |
| Flood Detection Accuracy | High      |

---

# Risk Analysis Engine

The system calculates victim danger levels using:

* Flood overlap ratio
* Local water density
* Body visibility
* Segmentation confidence

## Risk Levels

* LOW
* MEDIUM
* HIGH
* CRITICAL

---

# Streamlit Dashboard Features

## Dashboard Includes

* Futuristic AI UI
* Cyberpunk Design
* Glassmorphism Panels
* Flood Detection Feed
* Flood Overlay
* Binary Flood Mask
* Analytics Cards
* Risk Tables
* AI Monitoring Layout

---

# Tech Stack

## Programming

* Python

## Deep Learning

* PyTorch
* Ultralytics YOLOv8

## Computer Vision

* OpenCV
* NumPy

## UI

* Streamlit
* HTML/CSS

  <img width="1901" height="940" alt="image" src="https://github.com/user-attachments/assets/655bda0d-0711-43f2-ae85-ad64ce40e966" />



---

# Project Structure

```text
Flood-Victim-Detection/
│
├── classification/
├── configs/
├── data/
├── detection/
├── explainability/
├── models/
├── pipeline/
├── risk_engine/
├── segmentation/
├── splits/
├── streamlit_app/
├── tracking/
├── training/
├── utils/
├── video/
├── visualization/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Run Project

## 1. Clone Repository

```bash
git clone https://github.com/MdAshrafhussain889/Flood-Victim-Detection.git
```

---

## 2. Open Project

```bash
cd Flood-Victim-Detection
```

---

## 3. Create Virtual Environment

```bash
python -m venv venv
```

---

## 4. Activate Environment

### Windows

```bash
venv\Scripts\activate
```

---

## 5. Install Requirements

```bash
pip install -r requirements.txt
```

---

## 6. Run Streamlit App

```bash
streamlit run streamlit_app/app.py
```

---

# Input Modes

The system supports:

* Image Upload
* Video Upload
* Webcam Monitoring

---

# Future Enhancements

* Drone Integration
* Thermal Imaging
* GPS-Based Rescue Mapping
* Cloud Deployment
* Multi-Camera Monitoring
* Emergency Alert System
* Live AI Logs
* Real-Time FPS Optimization

---

# Author

Mohammed Ashraf Hussain

---

# License

This project is developed for research, academic, and AI disaster-management purposes.

```
```
