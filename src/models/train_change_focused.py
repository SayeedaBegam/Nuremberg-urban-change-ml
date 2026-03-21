"""Train models using change-focused features (T2 - T1) instead of raw spectral values."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.evaluate import regression_metrics
from src.models.uncertainty import random_forest_uncertainty
from src.utils.config import CHANGE_TARGET_COLUMNS, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE
from src.utils.io import ensure_directories, save_dataframe, save_model


def _spatial_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Approximate a spatial split using x-centroid median."""
    if "centroid_x_t1" in df.columns:
        cutoff = df["centroid_x_t1"].median()
        return df[df["centroid_x_t1"] <= cutoff], df[df["centroid_x_t1"] > cutoff]
    midpoint = len(df) // 2
    ordered = df.sort_values("cell_id").reset_index(drop=True)
    return ordered.iloc[:midpoint], ordered.iloc[midpoint:]


def _create_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create change-based features: T2 values minus T1 values."""
    features = df.copy()

    # Spectral band changes
    for band in ['b2', 'b3', 'b4', 'b8', 'b11', 'b12']:
        for stat in ['mean', 'std']:
            col_name = f"{band}_{stat}"
            if f"{col_name}_t1" in features.columns and f"{col_name}_t2" in features.columns:
                features[f"delta_{col_name}"] = features[f"{col_name}_t2"] - features[f"{col_name}_t1"]

    # Index changes
    for index in ['ndvi', 'ndbi', 'ndwi', 'brightness']:
        t1_col = f"{index}_mean_t1"
        t2_col = f"{index}_mean_t2"
        if t1_col in features.columns and t2_col in features.columns:
            features[f"delta_{index}_mean"] = features[t2_col] - features[t1_col]

    # Composition changes (already in dataset as delta_*)
    # But we'll also add ratio changes
    for lc_class in ['built_up', 'vegetation', 'water', 'other']:
        t1_col = f"{lc_class}_prop_t1"
        t2_col = f"{lc_class}_prop_t2"
        if t1_col in features.columns and t2_col in features.columns:
            # Add ratio of change relative to T1 value (avoid division by zero)
            features[f"ratio_{lc_class}_change"] = np.where(
                features[t1_col] > 0.01,
                features[f"delta_{lc_class}"] / features[t1_col],
                0.0
            )

    return features


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Get all change-based feature columns (excluding targets)."""
    excluded = {
        "cell_id",
        "year_label_t1",
        "year_label_t2",
        "change_binary",
        "centroid_x_t1",
        "centroid_y_t1",
        "centroid_x_t2",
        "centroid_y_t2",
        # Exclude land cover targets
        "delta_built_up",
        "delta_vegetation",
        "delta_water",
        "delta_other",
        "built_up_prop_t1",
        "vegetation_prop_t1",
        "water_prop_t1",
        "other_prop_t1",
        "built_up_prop_t2",
        "vegetation_prop_t2",
        "water_prop_t2",
        "other_prop_t2",
    }

    # Only keep spectral change features (delta_b*, delta_index*)
    features = [
        col for col in df.columns
        if (col.startswith("delta_") or col.startswith("ratio_"))
        and col not in excluded
    ]
    return features


def main() -> None:
    ensure_directories([MODELS_DIR])
    dataset = pd.read_csv(PROCESSED_DIR / "change_dataset.csv")

    # Create change-focused features
    print("Creating change-focused features...")
    dataset_with_changes = _create_change_features(dataset)

    train_df, test_df = _spatial_split(dataset_with_changes)

    feature_columns = _feature_columns(dataset_with_changes)
    print(f"Using {len(feature_columns)} change-based features")
    print(f"Features: {feature_columns[:10]}...")

    x_train = train_df[feature_columns]
    x_test = test_df[feature_columns]
    y_train = train_df[CHANGE_TARGET_COLUMNS]
    y_test = test_df[CHANGE_TARGET_COLUMNS]

    # Model 1: Elastic Net with scaling
    elastic_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                MultiOutputRegressor(
                    ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE),
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # Model 2: Random Forest (no scaling needed)
    forest_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                MultiOutputRegressor(
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=12,
                        min_samples_leaf=5,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("Training models...")
    elastic_model.fit(x_train, y_train)
    forest_model.fit(x_train, y_train)

    elastic_pred = pd.DataFrame(
        elastic_model.predict(x_test),
        columns=CHANGE_TARGET_COLUMNS,
        index=y_test.index
    )
    forest_pred = pd.DataFrame(
        forest_model.predict(x_test),
        columns=CHANGE_TARGET_COLUMNS,
        index=y_test.index
    )

    # Evaluate
    from sklearn.metrics import r2_score
    elastic_metrics = regression_metrics(y_test, elastic_pred)
    forest_metrics = regression_metrics(y_test, forest_pred)

    # Add R² scores
    for target in CHANGE_TARGET_COLUMNS:
        elastic_metrics[f"{target}_r2"] = float(r2_score(y_test[target], elastic_pred[target]))
        forest_metrics[f"{target}_r2"] = float(r2_score(y_test[target], forest_pred[target]))

    print("\n=== PERFORMANCE IMPROVEMENT ===")
    print(f"Elastic Net - Built-up R²: {elastic_metrics['delta_built_up_r2']:.4f}, MAE: {elastic_metrics['delta_built_up_mae']:.4f}")
    print(f"Random Forest - Built-up R²: {forest_metrics['delta_built_up_r2']:.4f}, MAE: {forest_metrics['delta_built_up_mae']:.4f}")

    # Compute uncertainty
    rf_model = forest_model.named_steps["model"]
    imputer = forest_model.named_steps["imputer"]
    x_test_imputed = imputer.transform(x_test)
    built_up_rf = rf_model.estimators_[0]
    built_up_uncertainty = random_forest_uncertainty(built_up_rf, x_test_imputed)

    # Export predictions
    prediction_export = test_df[["cell_id", "centroid_x_t1", "centroid_y_t1"]].copy()
    for target in CHANGE_TARGET_COLUMNS:
        prediction_export[f"actual_{target}"] = y_test[target].values
        prediction_export[f"pred_{target}"] = forest_pred[target].values
    prediction_export["uncertainty_built_up"] = built_up_uncertainty

    from src.utils.io import save_dataframe
    save_dataframe(prediction_export, PROCESSED_DIR / "app_predictions.csv")

    # Save models
    save_model(elastic_model, MODELS_DIR / "elastic_net_change_focused.joblib")
    save_model(forest_model, MODELS_DIR / "random_forest_change_focused.joblib")

    # Save metrics
    metrics = {
        "approach": "change_focused",
        "features_used": feature_columns,
        "n_features": len(feature_columns),
        "elastic_net": elastic_metrics,
        "random_forest": forest_metrics,
    }

    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n✅ Models trained and saved!")


if __name__ == "__main__":
    main()
