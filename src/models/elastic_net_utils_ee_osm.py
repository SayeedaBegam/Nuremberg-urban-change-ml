from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.elastic_net_config_ee_osm import RANDOM_STATE_EE_OSM


def build_elastic_net_pipeline_ee_osm(
    feature_columns: list[str],
    alpha: float,
    l1_ratio: float,
    random_state: int = RANDOM_STATE_EE_OSM,
) -> Pipeline:
    """Create a feature-selecting, scaling, multi-output Elastic Net pipeline."""
    selector = ColumnTransformer(
        transformers=[("feature_selector_ee_osm", "passthrough", feature_columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = MultiOutputRegressor(
        ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=random_state,
            max_iter=20000,
        )
    )
    return Pipeline(
        steps=[
            ("select_features_ee_osm", selector),
            ("scale_features_ee_osm", StandardScaler()),
            ("model_ee_osm", model),
        ]
    )


def export_coefficients_ee_osm(
    pipeline: Pipeline,
    feature_columns: list[str],
    target_columns: list[str],
) -> pd.DataFrame:
    """Extract per-target Elastic Net coefficients for interpretability."""
    model = pipeline.named_steps["model_ee_osm"]
    rows: list[dict[str, float | str]] = []

    for target_name, estimator in zip(target_columns, model.estimators_):
        for feature_name, coefficient in zip(feature_columns, estimator.coef_):
            rows.append(
                {
                    "target": target_name,
                    "feature": feature_name,
                    "coefficient": float(coefficient),
                    "abs_coefficient": float(abs(coefficient)),
                }
            )

    coefficient_frame = pd.DataFrame(rows)
    return coefficient_frame.sort_values(["target", "abs_coefficient"], ascending=[True, False]).reset_index(drop=True)
