from __future__ import annotations

from pathlib import Path


PROJECT_ROOT_EE = Path(__file__).resolve().parents[2]
DATA_DIR_EE = PROJECT_ROOT_EE / "data"
TRAIN_DATA_DIR_EE = DATA_DIR_EE / "train_esa_ee"
MODEL_OUTPUT_DIR_EE = PROJECT_ROOT_EE / "models" / "random_forest_ee"

TRAIN_DATA_PATH_EE = TRAIN_DATA_DIR_EE / "nuremberg_2019_to_2020_model.csv"
FORWARD_DATA_PATH_EE = TRAIN_DATA_DIR_EE / "nuremberg_2020_to_2021_model.csv"

FEATURE_BASE_COLUMNS_EE = [
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "NDVI",
    "NDWI",
    "NDBI",
]

TARGET_BASE_COLUMNS_EE = [
    "vegetation",
    "built_up",
    "water",
    "other",
]

TARGET_COLUMNS_EE = [f"{column}_t2" for column in TARGET_BASE_COLUMNS_EE]
FEATURE_COLUMNS_EE = [f"{column}_t1" for column in FEATURE_BASE_COLUMNS_EE]

VALIDATION_SIZE_EE = 0.10
TEST_SIZE_EE = 0.10
RANDOM_STATE_EE = 42

N_ESTIMATORS_GRID_EE = [100, 200, 300]
MAX_DEPTH_GRID_EE = [None, 10, 20]
MIN_SAMPLES_LEAF_GRID_EE = [1, 3, 5]
MAX_FEATURES_GRID_EE = ["sqrt", 0.8]
