# ============================================================
# data/create_splits.py
# Train/Validation/Test Split Generator
# ============================================================

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import (
    METADATA_CSV,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    SPLITS_ROOT,
)

df = pd.read_csv(METADATA_CSV)

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"],
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"],
)

os.makedirs(SPLITS_ROOT, exist_ok=True)
train_df.to_csv(TRAIN_CSV, index=False)
val_df.to_csv(VAL_CSV,   index=False)
test_df.to_csv(TEST_CSV,  index=False)

print("\nDataset Splits Created")
print(f"Train : {len(train_df)}")
print(f"Val   : {len(val_df)}")
print(f"Test  : {len(test_df)}")
