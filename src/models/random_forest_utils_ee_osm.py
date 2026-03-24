from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from src.utils.random_forest_config_ee_osm import RANDOM_STATE_EE_OSM


def build_random_forest_pipeline_ee_osm(
    feature_columns: list[str],
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    max_features: str | float,
    random_state: int = RANDOM_STATE_EE_OSM,
) -> Pipeline:
    """Create a feature-selecting, imputing, multi-output Random Forest pipeline."""
    selector = ColumnTransformer(
        transformers=[("feature_selector_ee_osm", "passthrough", feature_columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )
    )
    return Pipeline(
        steps=[
            ("select_features_ee_osm", selector),
            ("impute_features_ee_osm", SimpleImputer(strategy="mean")),
            ("model_ee_osm", model),
        ]
    )


def estimate_uncertainty_ee_osm(pipeline: Pipeline, dataframe: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    """Estimate per-target uncertainty as the std of tree predictions."""
    transformed_features = pipeline.named_steps["impute_features_ee_osm"].transform(
        pipeline.named_steps["select_features_ee_osm"].transform(dataframe)
    )
    model = pipeline.named_steps["model_ee_osm"]
    uncertainty_columns: dict[str, np.ndarray] = {}

    for target_name, estimator in zip(target_columns, model.estimators_):
        tree_predictions = np.stack([tree.predict(transformed_features) for tree in estimator.estimators_], axis=0)
        uncertainty_columns[f"uncertainty_{target_name}"] = tree_predictions.std(axis=0)

    uncertainty_frame = pd.DataFrame(uncertainty_columns, index=dataframe.index)
    uncertainty_frame["uncertainty_mean_row"] = uncertainty_frame.mean(axis=1)
    return uncertainty_frame


def export_feature_importances_ee_osm(
    pipeline: Pipeline,
    feature_columns: list[str],
    target_columns: list[str],
) -> pd.DataFrame:
    """Extract per-target feature importances from the fitted forest."""
    model = pipeline.named_steps["model_ee_osm"]
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
