from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.mlp_config_ee_osm import RANDOM_STATE_EE_OSM


def build_mlp_pipeline_ee_osm(
    feature_columns: list[str],
    hidden_layer_sizes: tuple[int, ...],
    activation: str,
    alpha: float,
    learning_rate_init: float,
    max_iter: int,
    random_state: int = RANDOM_STATE_EE_OSM,
) -> Pipeline:
    """Create a feature-selecting, imputing, scaling MLP regression pipeline."""
    selector = ColumnTransformer(
        transformers=[("feature_selector_ee_osm", "passthrough", feature_columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("select_features_ee_osm", selector),
            ("impute_features_ee_osm", SimpleImputer(strategy="mean")),
            ("scale_features_ee_osm", StandardScaler()),
            ("model_ee_osm", model),
        ]
    )
