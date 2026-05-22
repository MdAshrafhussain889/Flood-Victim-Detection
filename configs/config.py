# ============================================================
# configs/config.py
# Flood Victim Detection v3 - Central Configuration
# ============================================================

import os

# ============================================================
# ROOT PROJECT
# ============================================================
PROJECT_ROOT = r"C:\Users\Test1\Desktop\flood_victium_detection"

# ============================================================
# V3 PROJECT ROOT
# ============================================================
V3_ROOT = os.path.join(
    PROJECT_ROOT,
    "flood_victim_detection_v3",
)

# ============================================================
# ORIGINAL DATASETS
# ============================================================
DATASET1_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset 1",
)

DATASET2_ROOT = os.path.join(
    PROJECT_ROOT,
    "Dataset 2",
)

# ============================================================
# DATASET 1 - Segmentation Dataset
# ============================================================
DATASET1_IMAGES = os.path.join(
    DATASET1_ROOT,
    "images",
)

DATASET1_MASKS = os.path.join(
    DATASET1_ROOT,
    "masks",
)

# ============================================================
# DATASET 2 - Classification Dataset
# ============================================================
DATASET2_FLOOD = os.path.join(
    DATASET2_ROOT,
    "Flood Images",
)

DATASET2_NON_FLOOD = os.path.join(
    DATASET2_ROOT,
    "Non Flood Images",
)

# ============================================================
# LEGACY SINGLE DATASET PATHS
# (Backward Compatibility)
# ============================================================
IMAGE_DIR = DATASET1_IMAGES

MASK_DIR = DATASET1_MASKS

# ============================================================
# PROCESSED DATA
# ============================================================
PROCESSED_ROOT = os.path.join(
    V3_ROOT,
    "processed_data",
)

PROCESSED_IMAGES = os.path.join(
    PROCESSED_ROOT,
    "images",
)

PROCESSED_MASKS = os.path.join(
    PROCESSED_ROOT,
    "masks",
)

METADATA_CSV = os.path.join(
    PROCESSED_ROOT,
    "metadata.csv",
)

# ============================================================
# SPLITS
# ============================================================
SPLITS_ROOT = os.path.join(
    V3_ROOT,
    "splits",
)

TRAIN_CSV = os.path.join(
    SPLITS_ROOT,
    "train.csv",
)

VAL_CSV = os.path.join(
    SPLITS_ROOT,
    "val.csv",
)

TEST_CSV = os.path.join(
    SPLITS_ROOT,
    "test.csv",
)

# ============================================================
# IMAGE SETTINGS
# ============================================================
IMG_SIZE = 256

IMG_CHANNELS = 3

# ============================================================
# TRAINING SETTINGS
# ============================================================
SEED = 42

EPOCHS = 50

BATCH_SIZE = 8

LR = 1e-3

WEIGHT_DECAY = 1e-4

PATIENCE = 8

ACCUM_STEPS = 4

BCE_WEIGHT = 0.5

DICE_WEIGHT = 0.5

# ============================================================
# MODEL SELECTION
# ============================================================
MODEL_NAME = "attention_unet"

# ============================================================
# PATHS
# ============================================================
CHECKPOINT_DIR = os.path.join(
    V3_ROOT,
    "checkpoints",
)

OUTPUT_DIR = os.path.join(
    V3_ROOT,
    "outputs",
)

BEST_MODEL = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth",
)

# ============================================================
# DATALOADER
# ============================================================
TRAIN_RATIO = 0.70

VAL_RATIO = 0.15

TEST_RATIO = 0.15

NUM_WORKERS = 0

PIN_MEMORY = False

# ============================================================
# DEVICE
# ============================================================
DEVICE = "cpu"

# ============================================================
# CLASSIFICATION SETTINGS
# ============================================================
CLASSIFIER_IMG_SIZE = 224

CLASSIFIER_BATCH_SIZE = 32

CLASSIFIER_EPOCHS = 20

CLASSIFIER_LR = 1e-4

CLASSIFIER_WEIGHT_DECAY = 1e-4

CLASSIFIER_PATIENCE = 5

CLASSIFIER_THRESHOLD = 0.90

CLASSIFIER_MODEL_NAME = "efficientnet_b0"

CLASSIFIER_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "flood_classifier_best.pth",
)

# ============================================================
# SEGMENTATION SETTINGS
# ============================================================
SEGMENTATION_MODEL = "attention_unet"

SEGMENTATION_THRESHOLD = 0.50

MIN_FLOOD_CONFIDENCE = 0.15

MIN_FLOOD_PIXEL_RATIO = 0.005

MIN_REGION_AREA = 400

MORPH_KERNEL_SIZE = 5

USE_MORPHOLOGY = True

USE_CONFIDENCE_SUPPRESSION = True

# ============================================================
# RISK ENGINE SETTINGS
# ============================================================
RISK_LOW_THRESHOLD = 0.25

RISK_MEDIUM_THRESHOLD = 0.50

RISK_HIGH_THRESHOLD = 0.75

# ============================================================
# YOLO SETTINGS
# ============================================================
YOLO_MODEL = "yolov8n.pt"

YOLO_CONF = 0.40

YOLO_CONFIDENCE = 0.40

PERSON_CLASS_ID = 0

# ============================================================
# TRACKING SETTINGS
# ============================================================
ENABLE_TRACKING = True

TRACK_MAX_DISTANCE = 80

TRACK_MIN_CONFIDENCE = 0.30

# ============================================================
# VIDEO SETTINGS
# ============================================================
VIDEO_SKIP_FRAMES = 2

VIDEO_OUTPUT_FPS = 20

# ============================================================
# STREAMLIT SETTINGS
# ============================================================
APP_TITLE = "Flood Victim Detection v3"

# ============================================================
# AUTO CREATE IMPORTANT DIRECTORIES
# ============================================================
os.makedirs(PROCESSED_ROOT, exist_ok=True)

os.makedirs(PROCESSED_IMAGES, exist_ok=True)

os.makedirs(PROCESSED_MASKS, exist_ok=True)

os.makedirs(SPLITS_ROOT, exist_ok=True)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)




