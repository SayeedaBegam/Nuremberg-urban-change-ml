from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from src.models.evaluate import false_change_rate, regression_metrics, stability_score
from src.models.uncertainty import random_forest_uncertainty
from src.utils.config import CHANGE_TARGET_COLUMNS, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE
from src.utils.io import ensure_directories, save_dataframe, save_model


def _spatial_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Approximate a spatial split using x-centroid median from cell ids ordering fallback."""
    if "centroid_x_t1" in df.columns:
        cutoff = df["centroid_x_t1"].median()
        return df[df["centroid_x_t1"] <= cutoff], df[df["centroid_x_t1"] > cutoff]
    midpoint = len(df) // 2
    ordered = df.sort_values("cell_id").reset_index(drop=True)
    return ordered.iloc[:midpoint], ordered.iloc[midpoint:]


def _feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "cell_id",
        "year_label_t1",
        "year_label_t2",
        "change_binary",
        "centroid_x_t1",
        "centroid_y_t1",
        "centroid_x_t2",
        "centroid_y_t2",
        *CHANGE_TARGET_COLUMNS,
    }
    excluded.update(
        {
            "built_up_prop_t1",
            "vegetation_prop_t1",
            "water_prop_t1",
            "other_prop_t1",
            "built_up_prop_t2",
            "vegetation_prop_t2",
            "water_prop_t2",
            "other_prop_t2",
        }
    )
    return [column for column in df.columns if column not in excluded]


def main() -> None:
    ensure_directories([MODELS_DIR])
    dataset = pd.read_csv(PROCESSED_DIR / "change_dataset.csv")
    train_df, test_df = _spatial_split(dataset)

    feature_columns = _feature_columns(dataset)
    x_train = train_df[feature_columns]
    x_test = test_df[feature_columns]
    y_train = train_df[CHANGE_TARGET_COLUMNS]
    y_test = test_df[CHANGE_TARGET_COLUMNS]

    elastic_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                MultiOutputRegressor(
                    ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE),
                    n_jobs=1,
                ),
            ),
        ]
    )
    forest_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                MultiOutputRegressor(
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=14,
                        min_samples_leaf=3,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                    n_jobs=1,
                ),
            ),
        ]
    )

    elastic_model.fit(x_train, y_train)
    forest_model.fit(x_train, y_train)

    elastic_pred = pd.DataFrame(elastic_model.predict(x_test), columns=CHANGE_TARGET_COLUMNS, index=y_test.index)
    forest_pred = pd.DataFrame(forest_model.predict(x_test), columns=CHANGE_TARGET_COLUMNS, index=y_test.index)

    metrics = {
        "elastic": regression_metrics(y_test, elastic_pred),
        "forest": regression_metrics(y_test, forest_pred),
    }
    metrics["forest"]["delta_built_up_false_change_rate"] = false_change_rate(
        y_test["delta_built_up"], forest_pred["delta_built_up"]
    )
    metrics["forest"]["delta_built_up_stability"] = stability_score(
        y_test["delta_built_up"], forest_pred["delta_built_up"]
    )

    noisy_test = x_test.copy()
    noisy_test = noisy_test + np.random.normal(0, 0.05, size=noisy_test.shape)
    noisy_pred = pd.DataFrame(forest_model.predict(noisy_test), columns=CHANGE_TARGET_COLUMNS, index=y_test.index)
    metrics["forest_stress_test"] = regression_metrics(y_test, noisy_pred)

    rf_model = forest_model.named_steps["model"]
    imputer = forest_model.named_steps["imputer"]
    x_test_imputed = imputer.transform(x_test)
    built_up_rf = rf_model.estimators_[0]
    built_up_uncertainty = random_forest_uncertainty(built_up_rf, x_test_imputed)
    metrics["forest"]["mean_uncertainty"] = float(built_up_uncertainty.mean())

    prediction_export = test_df[["cell_id", "centroid_x_t1", "centroid_y_t1"]].copy()
    for target in CHANGE_TARGET_COLUMNS:
        prediction_export[f"actual_{target}"] = y_test[target].values
        prediction_export[f"pred_{target}"] = forest_pred[target].values
    prediction_export["uncertainty_built_up"] = built_up_uncertainty
    save_dataframe(prediction_export, PROCESSED_DIR / "app_predictions.csv")

    save_model(elastic_model, MODELS_DIR / "elastic_net.joblib")
    save_model(forest_model, MODELS_DIR / "random_forest.joblib")

    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


if __name__ == "__main__":
    main()
