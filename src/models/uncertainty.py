from __future__ import annotations

import numpy as np


def random_forest_uncertainty(model, features) -> np.ndarray:
    """Estimate uncertainty from disagreement between trees."""
    tree_predictions = np.stack([estimator.predict(features) for estimator in model.estimators_], axis=0)
    return tree_predictions.std(axis=0)
