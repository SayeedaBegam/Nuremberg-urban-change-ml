from __future__ import annotations

import importlib
import importlib.util

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from src.utils.xgboost_config_ee_osm import RANDOM_STATE_EE_OSM


def is_xgboost_available_ee_osm() -> bool:
    """Return True when xgboost is importable in the current environment."""
    return importlib.util.find_spec("xgboost") is not None


def get_xgboost_version_ee_osm() -> str | None:
    """Return the installed xgboost version, if available."""
    if not is_xgboost_available_ee_osm():
        return None
    module = importlib.import_module("xgboost")
    return getattr(module, "__version__", None)


def get_xgb_regressor_class_ee_osm():
    """Import and return XGBRegressor with a clear installation error if missing."""
    if not is_xgboost_available_ee_osm():
        raise ModuleNotFoundError(
            "xgboost is not installed in the current environment. "
            "Install it manually, for example with `pip install xgboost`, before running the XGBoost _ee_osm pipeline."
        )
    module = importlib.import_module("xgboost")
    return module.XGBRegressor


def build_xgboost_pipeline_ee_osm(
    feature_columns: list[str],
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    reg_alpha: float,
    reg_lambda: float,
    random_state: int = RANDOM_STATE_EE_OSM,
) -> Pipeline:
    """Create a feature-selecting, imputing, multi-output XGBoost pipeline."""
    XGBRegressor = get_xgb_regressor_class_ee_osm()
    selector = ColumnTransformer(
        transformers=[("feature_selector_ee_osm", "passthrough", feature_columns)],
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
            ("select_features_ee_osm", selector),
            ("impute_features_ee_osm", SimpleImputer(strategy="mean")),
            ("model_ee_osm", model),
        ]
    )


def export_feature_importances_ee_osm(
    pipeline: Pipeline,
    feature_columns: list[str],
    target_columns: list[str],
) -> pd.DataFrame:
    """Extract per-target feature importances from fitted XGBoost models."""
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
