from __future__ import annotations

import importlib
import importlib.util
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from src.utils.xgboost_config_ee import (
    FEATURE_BASE_COLUMNS_EE,
    FEATURE_COLUMNS_EE,
    RANDOM_STATE_EE,
    TARGET_BASE_COLUMNS_EE,
    TARGET_COLUMNS_EE,
    TEST_SIZE_EE,
    VALIDATION_SIZE_EE,
)


def is_xgboost_available_ee() -> bool:
    """Return True when xgboost is importable in the current environment."""
    return importlib.util.find_spec("xgboost") is not None


def get_xgboost_version_ee() -> str | None:
    """Return the installed xgboost version, if available."""
    if not is_xgboost_available_ee():
        return None
    module = importlib.import_module("xgboost")
    return getattr(module, "__version__", None)


def get_xgb_regressor_class_ee():
    """Import and return XGBRegressor with a clear installation error if missing."""
    if not is_xgboost_available_ee():
        raise ModuleNotFoundError(
            "xgboost is not installed in the current environment. "
            "Install it manually, for example with `pip install xgboost`, before running the XGBoost _ee pipeline."
        )
    module = importlib.import_module("xgboost")
    return module.XGBRegressor


def load_temporal_dataset_ee(path: Path) -> pd.DataFrame:
    """Load one temporal dataset and standardize feature and target names."""
    dataframe = pd.read_csv(path)
    if "cell_id" not in dataframe.columns:
        raise ValueError(f"{path} is missing the required 'cell_id' column.")

    feature_mapping = {}
    target_mapping = {}

    for base_name in FEATURE_BASE_COLUMNS_EE:
        matches = [column for column in dataframe.columns if re.fullmatch(rf"{base_name}_\d{{4}}", column)]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one feature column for {base_name} in {path}, found {matches}.")
        feature_mapping[matches[0]] = f"{base_name}_t1"

    for base_name in TARGET_BASE_COLUMNS_EE:
        matches = [column for column in dataframe.columns if re.fullmatch(rf"{base_name}_\d{{4}}", column)]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one target column for {base_name} in {path}, found {matches}.")
        target_mapping[matches[0]] = f"{base_name}_t2"

    standardized = dataframe.rename(columns={**feature_mapping, **target_mapping}).copy()
    required_columns = ["cell_id", *FEATURE_COLUMNS_EE, *TARGET_COLUMNS_EE]
    missing = [column for column in required_columns if column not in standardized.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns after renaming: {missing}")

    return standardized[required_columns]


def split_temporal_dataset_ee(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split one dataset into 80% train, 10% validation, and 10% test partitions."""
    train_validation_df, test_df = train_test_split(
        dataframe,
        test_size=TEST_SIZE_EE,
        random_state=RANDOM_STATE_EE,
        shuffle=True,
    )
    validation_fraction = VALIDATION_SIZE_EE / (1.0 - TEST_SIZE_EE)
    train_df, validation_df = train_test_split(
        train_validation_df,
        test_size=validation_fraction,
        random_state=RANDOM_STATE_EE,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), validation_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_xgboost_pipeline_ee(
    feature_columns: list[str],
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    reg_alpha: float,
    reg_lambda: float,
    random_state: int = RANDOM_STATE_EE,
) -> Pipeline:
    """Create a feature-selecting, imputing, multi-output XGBoost pipeline."""
    XGBRegressor = get_xgb_regressor_class_ee()
    selector = ColumnTransformer(
        transformers=[("feature_selector_ee", "passthrough", feature_columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = MultiOutputRegressor(
        XGBRegressor(
            objective="reg:squarederror",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=3,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="rmse",
            verbosity=0,
        )
    )
    return Pipeline(
        steps=[
            ("select_features_ee", selector),
            ("impute_features_ee", SimpleImputer(strategy="mean")),
            ("model_ee", model),
        ]
    )


def postprocess_predictions_ee(
    predictions: np.ndarray | pd.DataFrame,
    target_columns: list[str],
) -> pd.DataFrame:
    """Clip predictions to [0, 1] and normalize rows to sum to 1."""
    prediction_frame = pd.DataFrame(predictions, columns=target_columns).copy()
    clipped = prediction_frame.clip(lower=0.0, upper=1.0)
    row_sums = clipped.sum(axis=1)
    zero_sum_mask = row_sums <= 0.0
    if zero_sum_mask.any():
        clipped.loc[zero_sum_mask, :] = 1.0 / len(target_columns)
        row_sums = clipped.sum(axis=1)
    normalized = clipped.div(row_sums, axis=0)
    return normalized


def row_sum_summary_ee(frame: pd.DataFrame) -> dict[str, float]:
    """Summarize row-wise composition sums."""
    row_sums = frame.sum(axis=1)
    deviation = (row_sums - 1.0).abs()
    return {
        "row_sum_min": float(row_sums.min()),
        "row_sum_max": float(row_sums.max()),
        "row_sum_mean": float(row_sums.mean()),
        "max_abs_row_sum_error": float(deviation.max()),
        "mean_abs_row_sum_error": float(deviation.mean()),
        "zero_sum_rows": int((row_sums == 0.0).sum()),
    }


def compute_regression_metrics_ee(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    target_columns: list[str],
) -> dict[str, float]:
    """Compute per-target and overall regression metrics."""
    metrics: dict[str, float] = {}
    for column in target_columns:
        metrics[f"{column}_mae"] = float(mean_absolute_error(y_true[column], y_pred[column]))
        metrics[f"{column}_rmse"] = float(np.sqrt(mean_squared_error(y_true[column], y_pred[column])))
        metrics[f"{column}_r2"] = float(r2_score(y_true[column], y_pred[column]))

    true_array = y_true[target_columns].to_numpy()
    pred_array = y_pred[target_columns].to_numpy()
    metrics["overall_mae"] = float(mean_absolute_error(true_array.reshape(-1), pred_array.reshape(-1)))
    metrics["overall_rmse"] = float(np.sqrt(mean_squared_error(true_array.reshape(-1), pred_array.reshape(-1))))
    metrics["overall_r2"] = float(r2_score(true_array, pred_array, multioutput="uniform_average"))
    return metrics


def export_feature_importances_ee(
    pipeline: Pipeline,
    feature_columns: list[str],
    target_columns: list[str],
) -> pd.DataFrame:
    """Extract per-target feature importances from fitted XGBoost models."""
    model = pipeline.named_steps["model_ee"]
    rows: list[dict[str, float | str]] = []

    for target_name, estimator in zip(target_columns, model.estimators_):
        for feature_name, importance in zip(feature_columns, estimator.feature_importances_):
            rows.append(
                {
                    "target": target_name,
                    "feature": feature_name,
                    "importance": float(importance),
                }
            )

    importance_frame = pd.DataFrame(rows)
    return importance_frame.sort_values(["target", "importance"], ascending=[True, False]).reset_index(drop=True)


def build_prediction_export_ee(
    source_frame: pd.DataFrame,
    raw_predictions: np.ndarray | pd.DataFrame,
    normalized_predictions: pd.DataFrame,
    target_columns: list[str],
) -> pd.DataFrame:
    """Create a prediction export with actuals, raw predictions, and normalized predictions."""
    raw_frame = pd.DataFrame(raw_predictions, columns=target_columns, index=source_frame.index)
    export = pd.DataFrame({"cell_id": source_frame["cell_id"].values})

    for column in target_columns:
        export[f"actual_{column}"] = source_frame[column].values
        export[f"pred_raw_{column}"] = raw_frame[column].values
        export[f"pred_{column}"] = normalized_predictions[column].values

    export["pred_raw_row_sum"] = raw_frame.sum(axis=1).values
    export["pred_row_sum"] = normalized_predictions.sum(axis=1).values
    return export


def save_json_ee(payload: dict, path: Path) -> None:
    """Write a JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_model_ee(model: Pipeline, path: Path) -> None:
    """Persist the trained pipeline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
