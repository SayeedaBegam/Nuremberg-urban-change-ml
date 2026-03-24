from __future__ import annotations

from src.utils.common_config_ee_osm import (
    FEATURE_COLUMNS_EE_OSM,
    FINAL_TARGET_COLUMNS_EE_OSM,
    FORWARD_DATA_PATH_EE_OSM,
    MODELS_DIR_EE_OSM,
    RANDOM_STATE_EE_OSM,
    TEST_SIZE_EE_OSM,
    TRAIN_DATA_PATH_EE_OSM,
    TRAIN_TARGET_COLUMNS_EE_OSM,
    VALIDATION_SIZE_EE_OSM,
)


MODEL_OUTPUT_DIR_EE_OSM = MODELS_DIR_EE_OSM / "elastic_net_ee_osm"

ALPHA_GRID_EE_OSM = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
L1_RATIO_GRID_EE_OSM = [0.05, 0.2, 0.5, 0.8, 0.95]
