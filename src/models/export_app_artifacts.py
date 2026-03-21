from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_EE_DIR = DATA_DIR / "processed_esa_ee"
APP_PREDICTIONS_PATH = PROCESSED_DIR / "app_predictions.csv"
APP_METRICS_PATH = MODELS_DIR / "metrics.json"
LAND_COVER_CLASSES = ["built_up", "vegetation", "water", "other"]

MODEL_EXPORTS = {
    "elastic_net": {
        "prediction_files": {
            "test_2019_2020": MODELS_DIR / "elastic_net_ee" / "elastic_net_test_predictions_ee.csv",
            "forward_2020_2021": MODELS_DIR / "elastic_net_ee" / "elastic_net_external_predictions_ee.csv",
        },
        "metrics_file": MODELS_DIR / "elastic_net_ee" / "elastic_net_metrics_ee.json",
    },
    "random_forest": {
        "prediction_files": {
            "test_2019_2020": MODELS_DIR / "random_forest_ee" / "random_forest_test_predictions_ee.csv",
            "forward_2020_2021": MODELS_DIR / "random_forest_ee" / "random_forest_external_predictions_ee.csv",
        },
        "metrics_file": MODELS_DIR / "random_forest_ee" / "random_forest_metrics_ee.json",
    },
    "xgboost": {
        "prediction_files": {
            "test_2019_2020": MODELS_DIR / "xgboost_ee" / "xgboost_test_predictions_ee.csv",
            "forward_2020_2021": MODELS_DIR / "xgboost_ee" / "xgboost_external_predictions_ee.csv",
        },
        "metrics_file": MODELS_DIR / "xgboost_ee" / "xgboost_metrics_ee.json",
    },
    "mlp": {
        "prediction_files": {
            "test_2019_2020": MODELS_DIR / "mlp_ee" / "mlp_test_predictions_ee.csv",
            "forward_2020_2021": MODELS_DIR / "mlp_ee" / "mlp_external_predictions_ee.csv",
        },
        "metrics_file": MODELS_DIR / "mlp_ee" / "mlp_metrics_ee.json",
    },
}

T1_REFERENCE_PATHS = {
    "test_2019_2020": PROCESSED_EE_DIR / "nuremberg_2019_composition_250m.csv",
    "forward_2020_2021": PROCESSED_EE_DIR / "nuremberg_2020_composition_250m.csv",
}


def _load_t1_reference(split_name: str) -> pd.DataFrame:
    path = T1_REFERENCE_PATHS.get(split_name)
    if path is None or not path.exists():
        return pd.DataFrame({"cell_id": pd.Series(dtype="string")})

    reference = pd.read_csv(path)
    columns = ["cell_id", *LAND_COVER_CLASSES]
    missing = [column for column in columns if column not in reference.columns]
    if missing:
        return pd.DataFrame({"cell_id": pd.Series(dtype="string")})

    renamed = reference[columns].rename(columns={column: f"{column}_prop_t1" for column in LAND_COVER_CLASSES})
    renamed["cell_id"] = renamed["cell_id"].astype(str)
    return renamed


def _column_or_nan(frame: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name in frame.columns:
        return frame[column_name]
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def _standardize_prediction_export(model_name: str, split_name: str, path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    standardized = pd.DataFrame(
        {
            "cell_id": frame["cell_id"].astype(str),
            "model": model_name,
            "split": split_name,
            "pred_raw_row_sum": _column_or_nan(frame, "pred_raw_row_sum"),
            "pred_row_sum": _column_or_nan(frame, "pred_row_sum"),
            "uncertainty_mean_row": _column_or_nan(frame, "uncertainty_mean_row"),
        }
    )

    for class_name in LAND_COVER_CLASSES:
        standardized[f"pred_{class_name}_prop_t2"] = _column_or_nan(frame, f"pred_{class_name}_t2")
        standardized[f"actual_{class_name}_prop_t2"] = _column_or_nan(frame, f"actual_{class_name}_t2")
        standardized[f"uncertainty_{class_name}"] = _column_or_nan(frame, f"uncertainty_{class_name}_t2")

    merged = standardized.merge(_load_t1_reference(split_name), on="cell_id", how="left")
    expected_t1_columns = [f"{class_name}_prop_t1" for class_name in LAND_COVER_CLASSES]
    for column in expected_t1_columns:
        if column not in merged.columns:
            merged[column] = np.nan
    return merged


def _aggregate_metrics() -> dict:
    aggregated: dict[str, dict] = {}
    for model_name, paths in MODEL_EXPORTS.items():
        metrics_path = paths["metrics_file"]
        if not metrics_path.exists():
            continue

        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        aggregated[model_name] = {
            "best_params": payload.get("best_params_ee", {}),
            "feature_columns": payload.get("feature_columns_ee", []),
            "target_columns": payload.get("target_columns_ee", []),
            "row_counts": payload.get("row_counts_ee", {}),
            "metrics_by_split": {
                "test_2019_2020": payload.get("test_metrics_ee", {}),
                "forward_2020_2021": payload.get("external_forward_metrics_ee", {}),
            },
        }
    return aggregated


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    prediction_frames: list[pd.DataFrame] = []
    for model_name, paths in MODEL_EXPORTS.items():
        for split_name, prediction_path in paths["prediction_files"].items():
            if prediction_path.exists():
                prediction_frames.append(_standardize_prediction_export(model_name, split_name, prediction_path))

    if not prediction_frames:
        raise FileNotFoundError("No model prediction exports were found under models/*_ee/.")

    combined_predictions = pd.concat(prediction_frames, ignore_index=True)
    combined_predictions = combined_predictions.sort_values(["model", "split", "cell_id"]).reset_index(drop=True)
    combined_predictions.to_csv(APP_PREDICTIONS_PATH, index=False)

    aggregated_metrics = _aggregate_metrics()
    APP_METRICS_PATH.write_text(json.dumps(aggregated_metrics, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {APP_PREDICTIONS_PATH}")
    print(f"Wrote {APP_METRICS_PATH}")
    if not T1_REFERENCE_PATHS["test_2019_2020"].exists():
        print(
            "Note: 2019 composition was not found, so change mode for test_2019_2020 will be unavailable until "
            f"`{T1_REFERENCE_PATHS['test_2019_2020']}` exists."
        )


if __name__ == "__main__":
    main()
