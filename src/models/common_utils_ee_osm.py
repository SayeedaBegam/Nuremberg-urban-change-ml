from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.utils.common_config_ee_osm import (
    FEATURE_COLUMNS_EE_OSM,
    FINAL_TARGET_COLUMNS_EE_OSM,
    RANDOM_STATE_EE_OSM,
    TEST_SIZE_EE_OSM,
    TRAIN_TARGET_COLUMNS_EE_OSM,
    VALIDATION_SIZE_EE_OSM,
)


def load_temporal_dataset_ee_osm(path: Path) -> pd.DataFrame:
    """Load one temporal OSM dataset and validate the required schema."""
    dataframe = pd.read_csv(path)
    required_columns = ["cell_id", *FEATURE_COLUMNS_EE_OSM, *FINAL_TARGET_COLUMNS_EE_OSM]
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return dataframe[required_columns].copy()


def split_temporal_dataset_ee_osm(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the temporal dataset into 80% train, 10% validation, and 10% test."""
    train_validation_df, test_df = train_test_split(
        dataframe,
        test_size=TEST_SIZE_EE_OSM,
        random_state=RANDOM_STATE_EE_OSM,
        shuffle=True,
    )
    validation_fraction = VALIDATION_SIZE_EE_OSM / (1.0 - TEST_SIZE_EE_OSM)
    train_df, validation_df = train_test_split(
        train_validation_df,
        test_size=validation_fraction,
        random_state=RANDOM_STATE_EE_OSM,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), validation_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_feature_matrix_ee_osm(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return the fixed T1 feature matrix."""
    return dataframe[FEATURE_COLUMNS_EE_OSM].copy()


def get_training_targets_ee_osm(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return the three directly modeled targets."""
    return dataframe[TRAIN_TARGET_COLUMNS_EE_OSM].copy()


def get_final_targets_ee_osm(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return the four-target evaluation frame, including actual other."""
    return dataframe[FINAL_TARGET_COLUMNS_EE_OSM].copy()


def postprocess_three_target_predictions_ee_osm(
    predictions: np.ndarray | pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert raw 3-target predictions into valid 4-part compositions."""
    raw_frame = pd.DataFrame(predictions, columns=TRAIN_TARGET_COLUMNS_EE_OSM).copy()
    clipped = raw_frame.clip(lower=0.0, upper=1.0)
    sum3 = clipped.sum(axis=1)
    normalized_three = clipped.copy()
    overflow_mask = sum3 > 1.0
    if overflow_mask.any():
        normalized_three.loc[overflow_mask, :] = normalized_three.loc[overflow_mask, :].div(sum3[overflow_mask], axis=0)
    other = 1.0 - normalized_three.sum(axis=1)
    other = other.clip(lower=0.0, upper=1.0)

    final_frame = normalized_three.copy()
    final_frame["other"] = other
    final_frame = final_frame[FINAL_TARGET_COLUMNS_EE_OSM]

    final_row_sums = final_frame.sum(axis=1)
    if not np.allclose(final_row_sums.to_numpy(), 1.0, atol=1e-8):
        final_frame = final_frame.div(final_row_sums.replace(0.0, np.nan), axis=0).fillna(0.0)
        final_frame["other"] = 1.0 - final_frame[TRAIN_TARGET_COLUMNS_EE_OSM].sum(axis=1)
        final_frame["other"] = final_frame["other"].clip(lower=0.0, upper=1.0)
    return raw_frame, final_frame


def three_target_sum_summary_ee_osm(frame: pd.DataFrame) -> dict[str, float]:
    """Summarize the raw three-target sum before other is derived."""
    row_sums = frame.sum(axis=1)
    return {
        "sum3_min": float(row_sums.min()),
        "sum3_max": float(row_sums.max()),
        "sum3_mean": float(row_sums.mean()),
        "sum3_gt_1_rows": int((row_sums > 1.0).sum()),
        "sum3_le_0_rows": int((row_sums <= 0.0).sum()),
    }


def row_sum_summary_ee_osm(frame: pd.DataFrame) -> dict[str, float]:
    """Summarize how close final four-part outputs are to summing to one."""
    row_sums = frame[FINAL_TARGET_COLUMNS_EE_OSM].sum(axis=1)
    deviation = (row_sums - 1.0).abs()
    return {
        "row_sum_min": float(row_sums.min()),
        "row_sum_max": float(row_sums.max()),
        "row_sum_mean": float(row_sums.mean()),
        "max_abs_row_sum_error": float(deviation.max()),
        "mean_abs_row_sum_error": float(deviation.mean()),
        "zero_sum_rows": int((row_sums == 0.0).sum()),
    }


def compute_regression_metrics_ee_osm(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
) -> dict[str, float]:
    """Compute per-target and overall regression metrics for the final four outputs."""
    metrics: dict[str, float] = {}
    for column in FINAL_TARGET_COLUMNS_EE_OSM:
        metrics[f"{column}_mae"] = float(mean_absolute_error(y_true[column], y_pred[column]))
        metrics[f"{column}_rmse"] = float(np.sqrt(mean_squared_error(y_true[column], y_pred[column])))
        metrics[f"{column}_r2"] = float(r2_score(y_true[column], y_pred[column]))

    true_array = y_true[FINAL_TARGET_COLUMNS_EE_OSM].to_numpy()
    pred_array = y_pred[FINAL_TARGET_COLUMNS_EE_OSM].to_numpy()
    metrics["overall_mae"] = float(mean_absolute_error(true_array.reshape(-1), pred_array.reshape(-1)))
    metrics["overall_rmse"] = float(np.sqrt(mean_squared_error(true_array.reshape(-1), pred_array.reshape(-1))))
    metrics["overall_r2"] = float(r2_score(true_array, pred_array, multioutput="uniform_average"))
    return metrics


def build_prediction_export_ee_osm(
    source_frame: pd.DataFrame,
    raw_three_predictions: pd.DataFrame,
    final_predictions: pd.DataFrame,
    model_name: str,
    split_name: str,
    uncertainty_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create a prediction export with actuals, predictions, residuals, and diagnostics."""
    export = pd.DataFrame(
        {
            "cell_id": source_frame["cell_id"].values,
            "model": model_name,
            "split": split_name,
        }
    )
    actuals = get_final_targets_ee_osm(source_frame)

    for column in FINAL_TARGET_COLUMNS_EE_OSM:
        export[f"actual_{column}"] = actuals[column].values
        export[f"pred_{column}"] = final_predictions[column].values
        export[f"residual_{column}"] = final_predictions[column].values - actuals[column].values
        export[f"abs_error_{column}"] = np.abs(export[f"residual_{column}"])

    for column in TRAIN_TARGET_COLUMNS_EE_OSM:
        export[f"pred_raw_{column}"] = raw_three_predictions[column].values

    export["actual_row_sum"] = actuals.sum(axis=1).values
    export["pred_raw_sum3"] = raw_three_predictions.sum(axis=1).values
    export["pred_row_sum"] = final_predictions.sum(axis=1).values
    export["abs_error_mean"] = export[[f"abs_error_{column}" for column in FINAL_TARGET_COLUMNS_EE_OSM]].mean(axis=1)

    if uncertainty_frame is not None:
        for column in uncertainty_frame.columns:
            export[column] = uncertainty_frame[column].values

    return export


def save_json_ee_osm(payload: dict, path: Path) -> None:
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_model_ee_osm(model, path: Path) -> None:
    """Persist a fitted model pipeline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
