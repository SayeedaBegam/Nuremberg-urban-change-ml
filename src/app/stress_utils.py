from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.common_utils_ee_osm import (
    FINAL_TARGET_COLUMNS_EE_OSM,
    FEATURE_COLUMNS_EE_OSM,
    get_final_targets_ee_osm,
    load_temporal_dataset_ee_osm,
    postprocess_three_target_predictions_ee_osm,
    split_temporal_dataset_ee_osm,
)
from src.models.elastic_net_utils_ee import (
    FEATURE_COLUMNS_EE,
    TARGET_COLUMNS_EE,
    load_temporal_dataset_ee,
    postprocess_predictions_ee,
    split_temporal_dataset_ee,
)


COMMON_TARGET_ORDER = ["built_up", "vegetation", "water", "other"]
EE_TARGET_TO_COMMON = {
    "built_up_t2": "built_up",
    "vegetation_t2": "vegetation",
    "water_t2": "water",
    "other_t2": "other",
}


@lru_cache(maxsize=8)
def load_model_pipeline(model_path: str):
    return joblib.load(model_path)


@lru_cache(maxsize=8)
def load_dataset_for_split(pipeline_key: str, split_name: str, train_path: str, forward_path: str) -> pd.DataFrame:
    if pipeline_key == "ee":
        if split_name == "forward_2020_2021":
            return load_temporal_dataset_ee(Path(forward_path)).copy()
        training_frame = load_temporal_dataset_ee(Path(train_path))
        _, _, test_df = split_temporal_dataset_ee(training_frame)
        return test_df.copy()

    if split_name == "forward_2020_2021":
        return load_temporal_dataset_ee_osm(Path(forward_path)).copy()
    training_frame = load_temporal_dataset_ee_osm(Path(train_path))
    _, _, test_df = split_temporal_dataset_ee_osm(training_frame)
    return test_df.copy()


def feature_columns_for_pipeline(pipeline_key: str) -> list[str]:
    return FEATURE_COLUMNS_EE if pipeline_key == "ee" else FEATURE_COLUMNS_EE_OSM


def _actual_targets_common(pipeline_key: str, frame: pd.DataFrame) -> pd.DataFrame:
    if pipeline_key == "ee":
        actual = frame[TARGET_COLUMNS_EE].rename(columns=EE_TARGET_TO_COMMON).copy()
        return actual[COMMON_TARGET_ORDER]
    return get_final_targets_ee_osm(frame)[COMMON_TARGET_ORDER].copy()


def _predict_common(pipeline_key: str, model, frame: pd.DataFrame) -> pd.DataFrame:
    raw_predictions = model.predict(frame)
    if pipeline_key == "ee":
        normalized = postprocess_predictions_ee(raw_predictions, TARGET_COLUMNS_EE)
        prediction_frame = normalized.rename(columns=EE_TARGET_TO_COMMON).copy()
        return prediction_frame[COMMON_TARGET_ORDER]
    _, final_predictions = postprocess_three_target_predictions_ee_osm(raw_predictions)
    return final_predictions[COMMON_TARGET_ORDER].copy()


def apply_feature_noise(
    frame: pd.DataFrame,
    feature_columns: list[str],
    noise_level: float,
    noise_mode: str,
    selected_feature: str | None,
    random_state: int = 42,
) -> pd.DataFrame:
    noisy = frame.copy()
    if noise_level <= 0:
        return noisy

    rng = np.random.default_rng(random_state)
    target_features = feature_columns if noise_mode == "All features" else [selected_feature] if selected_feature else []

    for feature_name in target_features:
        if feature_name not in noisy.columns:
            continue
        feature_std = float(noisy[feature_name].std(ddof=0))
        if not np.isfinite(feature_std) or feature_std <= 0:
            continue
        noise = rng.normal(loc=0.0, scale=noise_level * feature_std, size=len(noisy))
        noisy[feature_name] = noisy[feature_name].to_numpy(dtype=float) + noise
    return noisy


def compute_regression_metrics_common(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict[str, float]:
    true_array = y_true[COMMON_TARGET_ORDER].to_numpy()
    pred_array = y_pred[COMMON_TARGET_ORDER].to_numpy()
    return {
        "mae": float(mean_absolute_error(true_array.reshape(-1), pred_array.reshape(-1))),
        "rmse": float(np.sqrt(mean_squared_error(true_array.reshape(-1), pred_array.reshape(-1)))),
        "r2": float(r2_score(true_array, pred_array, multioutput="uniform_average")),
    }


def run_stress_test(
    pipeline_key: str,
    model_path: Path,
    train_data_path: Path,
    forward_data_path: Path,
    split_name: str,
    noise_level: float,
    noise_mode: str,
    selected_feature: str | None,
) -> dict[str, object]:
    model = load_model_pipeline(str(model_path))
    dataset = load_dataset_for_split(pipeline_key, split_name, str(train_data_path), str(forward_data_path)).copy()
    feature_columns = feature_columns_for_pipeline(pipeline_key)

    original_predictions = _predict_common(pipeline_key, model, dataset)
    noisy_dataset = apply_feature_noise(dataset, feature_columns, noise_level, noise_mode, selected_feature)
    noisy_predictions = _predict_common(pipeline_key, model, noisy_dataset)
    actual_targets = _actual_targets_common(pipeline_key, dataset)

    shift_frame = (noisy_predictions - original_predictions).abs()
    shift_frame = shift_frame.rename(columns={class_name: f"prediction_shift_{class_name}" for class_name in COMMON_TARGET_ORDER})
    shift_frame["prediction_shift_mean"] = shift_frame.mean(axis=1)

    result_frame = pd.DataFrame({"cell_id": dataset["cell_id"].astype(str).values})
    for class_name in COMMON_TARGET_ORDER:
        result_frame[f"actual_{class_name}_prop_t2"] = actual_targets[class_name].values
        result_frame[f"pred_{class_name}_prop_t2"] = original_predictions[class_name].values
        result_frame[f"pred_noisy_{class_name}_prop_t2"] = noisy_predictions[class_name].values
        result_frame[f"prediction_shift_{class_name}"] = shift_frame[f"prediction_shift_{class_name}"].values
    result_frame["prediction_shift_mean"] = shift_frame["prediction_shift_mean"].values

    return {
        "result_frame": result_frame,
        "feature_columns": feature_columns,
        "metrics_before": compute_regression_metrics_common(actual_targets, original_predictions),
        "metrics_after": compute_regression_metrics_common(actual_targets, noisy_predictions),
        "mean_shift": float(shift_frame["prediction_shift_mean"].mean()),
        "max_shift": float(shift_frame["prediction_shift_mean"].max()),
    }
