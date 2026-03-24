from __future__ import annotations

import numpy as np


def random_forest_uncertainty(model, features) -> np.ndarray:
    """Estimate uncertainty from disagreement between trees."""
    tree_predictions = np.stack([estimator.predict(features) for estimator in model.estimators_], axis=0)
    return tree_predictions.std(axis=0)


def elastic_net_uncertainty(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Estimate uncertainty for elastic net based on residual magnitude.
    Uses absolute residuals as a proxy for prediction uncertainty.
    Normalized by the scale of the target values to make it interpretable.
    """
    residuals = np.abs(y_true - y_pred)
    
    # Use a rolling window approach: normalize each residual by local context
    # to capture relative uncertainty rather than just raw error magnitude
    if len(residuals) == 0:
        return residuals
    
    # Compute percentile-based normalization to handle outliers
    percentile_95 = np.percentile(residuals, 95)
    if percentile_95 > 0:
        normalized_uncertainty = residuals / percentile_95
        return np.clip(normalized_uncertainty, 0, 1)  # Clip to [0, 1] range
    
    return residuals
