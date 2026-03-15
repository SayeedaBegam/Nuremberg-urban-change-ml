from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict[str, float]:
    metrics = {}
    for column in y_true.columns:
        metrics[f"{column}_mae"] = float(mean_absolute_error(y_true[column], y_pred[column]))
        metrics[f"{column}_rmse"] = float(np.sqrt(mean_squared_error(y_true[column], y_pred[column])))
    return metrics


def false_change_rate(y_true: pd.Series, y_pred: pd.Series, threshold: float = 0.1) -> float:
    actual_stable = y_true.abs() < threshold
    predicted_change = y_pred.abs() >= threshold
    if actual_stable.sum() == 0:
        return 0.0
    return float((predicted_change & actual_stable).sum() / actual_stable.sum())


def stability_score(y_true: pd.Series, y_pred: pd.Series, threshold: float = 0.05) -> float:
    stable_mask = y_true.abs() < threshold
    if stable_mask.sum() == 0:
        return 0.0
    stable_error = np.abs(y_true[stable_mask] - y_pred[stable_mask])
    return float(1.0 - stable_error.mean())
