from __future__ import annotations

from pathlib import Path


PROJECT_ROOT_EE_OSM = Path(__file__).resolve().parents[2]
DATA_DIR_EE_OSM = PROJECT_ROOT_EE_OSM / "data"
TRAIN_DATA_DIR_EE_OSM = DATA_DIR_EE_OSM / "train_esa_osm"
MODELS_DIR_EE_OSM = PROJECT_ROOT_EE_OSM / "models"

TRAIN_DATA_PATH_EE_OSM = TRAIN_DATA_DIR_EE_OSM / "nuremberg_2019_to_2020_model_osm.csv"
FORWARD_DATA_PATH_EE_OSM = TRAIN_DATA_DIR_EE_OSM / "nuremberg_2020_to_2021_model_osm.csv"

FEATURE_COLUMNS_EE_OSM = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "NDVI",
    "NDWI",
    "NDBI",
    "MNDWI",
    "BSI",
    "brightness",
    "NDVI_std",
    "NDBI_std",
    "distance_to_boundary_centroid",
    "road_density",
    "building_area_ratio",
]

TRAIN_TARGET_COLUMNS_EE_OSM = [
    "vegetation",
    "built_up",
    "water",
]

FINAL_TARGET_COLUMNS_EE_OSM = [
    "vegetation",
    "built_up",
    "water",
    "other",
]

EXCLUDED_COLUMNS_EE_OSM = [
    "cell_id",
    "system:index",
    ".geo",
]

VALIDATION_SIZE_EE_OSM = 0.10
TEST_SIZE_EE_OSM = 0.10
RANDOM_STATE_EE_OSM = 42
