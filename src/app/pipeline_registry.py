from __future__ import annotations

from pathlib import Path

from src.utils.config import INTERIM_DIR, MODELS_DIR, PROCESSED_DIR
from src.utils.elastic_net_config_ee import FORWARD_DATA_PATH_EE, TRAIN_DATA_PATH_EE
from src.utils.common_config_ee_osm import FORWARD_DATA_PATH_EE_OSM, TRAIN_DATA_PATH_EE_OSM


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_EE_DIR = PROJECT_ROOT / "data" / "processed_esa_ee"
PROCESSED_EE_OSM_DIR = PROJECT_ROOT / "data" / "processed_esa_ee_osm"

PIPELINE_REGISTRY = {
    "ee": {
        "label": "Sentinel-2",
        "grid_path": INTERIM_DIR / "grid.geojson",
        "grid_fallback_paths": [
            PROCESSED_EE_DIR / "nuremberg_2020_composition_250m.csv",
            PROCESSED_EE_DIR / "nuremberg_2021_composition_250m.csv",
            PROCESSED_EE_DIR / "nuremberg_2019_features_250m.csv",
        ],
        "predictions_path": PROCESSED_DIR / "app_predictions.csv",
        "metrics_path": MODELS_DIR / "metrics.json",
        "elastic_net_coefficients_path": MODELS_DIR / "elastic_net_ee" / "elastic_net_coefficients_ee.csv",
        "model_artifact_paths": {
            "elastic_net": MODELS_DIR / "elastic_net_ee" / "elastic_net_model_ee.joblib",
            "random_forest": MODELS_DIR / "random_forest_ee" / "random_forest_model_ee.joblib",
            "xgboost": MODELS_DIR / "xgboost_ee" / "xgboost_model_ee.joblib",
            "mlp": MODELS_DIR / "mlp_ee" / "mlp_model_ee.joblib",
        },
        "dataset_paths_by_split": {
            "test_2019_2020": TRAIN_DATA_PATH_EE,
            "forward_2020_2021": FORWARD_DATA_PATH_EE,
        },
        "comparison_summary_path": None,
        "supported_models": ["elastic_net", "random_forest", "xgboost", "mlp"],
        "supported_splits": ["test_2019_2020", "forward_2020_2021"],
        "sidebar_note": None,
        "t1_composition_by_split": {},
    },
    "ee_osm": {
        "label": "Sentinel-2 + OSM",
        "grid_path": PROCESSED_EE_OSM_DIR / "nuremberg_grid_250m_wgs84.geojson",
        "grid_fallback_paths": [
            PROCESSED_EE_OSM_DIR / "nuremberg_2020_composition_250m.csv",
            PROCESSED_EE_OSM_DIR / "nuremberg_2021_composition_250m.csv",
            PROCESSED_EE_OSM_DIR / "nuremberg_2019_osm_features_250m.csv",
        ],
        "prediction_exports": {
            "elastic_net": {
                "test_2019_2020": MODELS_DIR / "elastic_net_ee_osm" / "elastic_net_test_predictions_ee_osm.csv",
                "forward_2020_2021": MODELS_DIR / "elastic_net_ee_osm" / "elastic_net_external_predictions_ee_osm.csv",
            },
            "random_forest": {
                "test_2019_2020": MODELS_DIR / "random_forest_ee_osm" / "random_forest_test_predictions_ee_osm.csv",
                "forward_2020_2021": MODELS_DIR / "random_forest_ee_osm" / "random_forest_external_predictions_ee_osm.csv",
            },
            "xgboost": {
                "test_2019_2020": MODELS_DIR / "xgboost_ee_osm" / "xgboost_test_predictions_ee_osm.csv",
                "forward_2020_2021": MODELS_DIR / "xgboost_ee_osm" / "xgboost_external_predictions_ee_osm.csv",
            },
            "mlp": {
                "test_2019_2020": MODELS_DIR / "mlp_ee_osm" / "mlp_test_predictions_ee_osm.csv",
                "forward_2020_2021": MODELS_DIR / "mlp_ee_osm" / "mlp_external_predictions_ee_osm.csv",
            },
        },
        "metrics_exports": {
            "elastic_net": MODELS_DIR / "elastic_net_ee_osm" / "elastic_net_metrics_ee_osm.json",
            "random_forest": MODELS_DIR / "random_forest_ee_osm" / "random_forest_metrics_ee_osm.json",
            "xgboost": MODELS_DIR / "xgboost_ee_osm" / "xgboost_metrics_ee_osm.json",
            "mlp": MODELS_DIR / "mlp_ee_osm" / "mlp_metrics_ee_osm.json",
        },
        "elastic_net_coefficients_path": MODELS_DIR / "elastic_net_ee_osm" / "elastic_net_coefficients_ee_osm.csv",
        "model_artifact_paths": {
            "elastic_net": MODELS_DIR / "elastic_net_ee_osm" / "elastic_net_model_ee_osm.joblib",
            "random_forest": MODELS_DIR / "random_forest_ee_osm" / "random_forest_model_ee_osm.joblib",
            "xgboost": MODELS_DIR / "xgboost_ee_osm" / "xgboost_model_ee_osm.joblib",
            "mlp": MODELS_DIR / "mlp_ee_osm" / "mlp_model_ee_osm.joblib",
        },
        "dataset_paths_by_split": {
            "test_2019_2020": TRAIN_DATA_PATH_EE_OSM,
            "forward_2020_2021": FORWARD_DATA_PATH_EE_OSM,
        },
        "comparison_summary_path": MODELS_DIR / "model_comparison_ee_osm" / "model_comparison_summary_ee_osm.csv",
        "supported_models": ["elastic_net", "random_forest", "xgboost", "mlp"],
        "supported_splits": ["test_2019_2020", "forward_2020_2021"],
        "sidebar_note": "Auxiliary OSM features enabled",
        "t1_composition_by_split": {
            "forward_2020_2021": PROCESSED_EE_OSM_DIR / "nuremberg_2020_composition_250m.csv",
        },
    },
}

PIPELINE_ORDER = ["ee", "ee_osm"]
PIPELINE_LABELS = [PIPELINE_REGISTRY[key]["label"] for key in PIPELINE_ORDER]
PIPELINE_LABEL_TO_KEY = {PIPELINE_REGISTRY[key]["label"]: key for key in PIPELINE_ORDER}
