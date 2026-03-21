"""Train enhanced models using 2020→2021 data with improved features for urban change detection."""

from __future__ import annotations

import json
import logging

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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _create_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create enhanced change-based features optimized for 2020-2021 urban change detection."""
    features = df.copy()

    # Basic spectral band changes
    for band in ['b2', 'b3', 'b4', 'b8', 'b11', 'b12']:
        for stat in ['mean', 'std']:
            col_name = f"{band}_{stat}"
            if f"{col_name}_t1" in features.columns and f"{col_name}_t2" in features.columns:
                features[f"delta_{col_name}"] = features[f"{col_name}_t2"] - features[f"{col_name}_t1"]

                # Add ratio change (normalized by T1 to capture relative magnitude)
                t1_vals = features[f"{col_name}_t1"]
                t2_vals = features[f"{col_name}_t2"]
                features[f"ratio_{col_name}"] = np.where(
                    t1_vals.abs() > 1e-5,
                    (t2_vals - t1_vals) / (t1_vals.abs() + 1e-3),
                    0.0
                )

    # Spectral indices changes
    for index in ['ndvi', 'ndbi', 'ndwi', 'brightness']:
        t1_col = f"{index}_mean_t1"
        t2_col = f"{index}_mean_t2"
        if t1_col in features.columns and t2_col in features.columns:
            features[f"delta_{index}_mean"] = features[t2_col] - features[t1_col]

            # Absolute change (magnitude regardless of direction)
            features[f"abs_delta_{index}_mean"] = features[f"delta_{index}_mean"].abs()

    # Land cover proportion changes
    for lc_class in ['built_up', 'vegetation', 'water', 'other']:
        t1_col = f"{lc_class}_prop_t1"
        t2_col = f"{lc_class}_prop_t2"
        delta_col = f"delta_{lc_class}"
        if delta_col in features.columns:
            features[f"abs_delta_{lc_class}"] = features[delta_col].abs()

            # Relative change (ratio)
            if t1_col in features.columns:
                features[f"ratio_{lc_class}_change"] = np.where(
                    features[t1_col] > 0.01,
                    features[delta_col] / features[t1_col],
                    0.0
                )

    # Feature interactions (capture complex urban patterns)
    # Urbanization intensity: change in built-up driven by vegetation loss
    if "delta_built_up" in features.columns and "delta_vegetation" in features.columns:
        features["urbanization_intensity"] = (
            features["delta_built_up"].clip(lower=0) *
            (-features["delta_vegetation"]).clip(lower=0)
        )

    # Vegetation loss magnitude
    if "delta_vegetation" in features.columns:
        features["vegetation_loss_magnitude"] = (-features["delta_vegetation"]).clip(lower=0)

    # Built-up addition magnitude
    if "delta_built_up" in features.columns:
        features["built_up_gain_magnitude"] = features["delta_built_up"].clip(lower=0)

    # NDVI-NDBI divergence (green spaces vs built-up)
    if "delta_ndvi_mean" in features.columns and "delta_ndbi_mean" in features.columns:
        features["ndvi_ndbi_divergence"] = (
            features["delta_ndvi_mean"] - features["delta_ndbi_mean"]
        )

    # Spectral diversity (std of band changes)
    band_deltas = [col for col in features.columns if col.startswith("delta_b") and col.endswith("_mean")]
    if band_deltas:
        features["spectral_change_diversity"] = features[[f"abs_{col}" if f"abs_{col}" in features.columns else col for col in band_deltas]].std(axis=1)

    return features


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Get all enhanced change-based feature columns."""
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
        "ndvi_std_t1",
        "ndvi_std_t2",
        "ndbi_mean_t1",
        "ndbi_mean_t2",
    }

    features = [
        col for col in df.columns
        if col not in excluded and not col.startswith("centroid")
    ]
    return features


def main() -> None:
    ensure_directories([MODELS_DIR])

    # Load 2020-2021 change dataset
    change_dataset_path = PROCESSED_DIR / "change_dataset_2020_2021.csv"
    if not change_dataset_path.exists():
        logger.error(f"Change dataset not found: {change_dataset_path}")
        logger.info("Falling back to general change_dataset.csv")
        change_dataset_path = PROCESSED_DIR / "change_dataset.csv"

    dataset = pd.read_csv(change_dataset_path)
    logger.info(f"Loaded {len(dataset)} samples from {change_dataset_path}")

    # Create enhanced features
    logger.info("Creating enhanced features for 2020-2021 urban change...")
    dataset_with_changes = _create_enhanced_features(dataset)

    # Use random split for 2020-2021 (temporal data is already from same period)
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        dataset_with_changes,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    feature_columns = _feature_columns(dataset_with_changes)
    logger.info(f"Using {len(feature_columns)} enhanced features")
    logger.info(f"Features: {feature_columns[:15]}...")

    x_train = train_df[feature_columns].fillna(0)
    x_test = test_df[feature_columns].fillna(0)
    y_train = train_df[CHANGE_TARGET_COLUMNS]
    y_test = test_df[CHANGE_TARGET_COLUMNS]

    # Model 1: Elastic Net with enhanced regularization
    logger.info("Training Elastic Net...")
    elastic_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                MultiOutputRegressor(
                    ElasticNet(
                        alpha=0.005,  # Lower for 2020-2021 (smaller temporal gap)
                        l1_ratio=0.7,  # More L1 for feature selection
                        max_iter=5000,
                        random_state=RANDOM_STATE
                    ),
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # Model 2: Random Forest with optimized hyperparameters
    logger.info("Training Random Forest...")
    forest_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                MultiOutputRegressor(
                    RandomForestRegressor(
                        n_estimators=300,  # More trees for complex patterns
                        max_depth=15,
                        min_samples_leaf=3,
                        min_impurity_decrease=0.0001,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                    n_jobs=-1,
                ),
            ),
        ]
    )

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

    logger.info("\n" + "="*50)
    logger.info("2020-2021 ENHANCED MODEL PERFORMANCE")
    logger.info("="*50)
    logger.info(f"Elastic Net - Built-up R²: {elastic_metrics['delta_built_up_r2']:.4f}, MAE: {elastic_metrics['delta_built_up_mae']:.4f}")
    logger.info(f"Random Forest - Built-up R²: {forest_metrics['delta_built_up_r2']:.4f}, MAE: {forest_metrics['delta_built_up_mae']:.4f}")
    logger.info(f"Vegetation:")
    logger.info(f"  Elastic Net - R²: {elastic_metrics['delta_vegetation_r2']:.4f}, MAE: {elastic_metrics['delta_vegetation_mae']:.4f}")
    logger.info(f"  Random Forest - R²: {forest_metrics['delta_vegetation_r2']:.4f}, MAE: {forest_metrics['delta_vegetation_mae']:.4f}")
    logger.info("="*50)

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
    prediction_export["model"] = "random_forest"
    prediction_export["split"] = "test_2020_2021"

    save_dataframe(prediction_export, PROCESSED_DIR / "app_predictions_2020_2021.csv")
    logger.info(f"✅ Predictions exported to app_predictions_2020_2021.csv")

    # Save models
    save_model(elastic_model, MODELS_DIR / "elastic_net_2020_2021_enhanced.joblib")
    save_model(forest_model, MODELS_DIR / "random_forest_2020_2021_enhanced.joblib")
    logger.info("✅ Models saved")

    # Save metrics
    metrics = {
        "approach": "2020_2021_enhanced",
        "description": "Enhanced features for 2020-2021 urban change detection",
        "features_used": feature_columns,
        "n_features": len(feature_columns),
        "elastic_net": elastic_metrics,
        "random_forest": forest_metrics,
    }

    with open(MODELS_DIR / "metrics_2020_2021_enhanced.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("✅ Metrics saved to metrics_2020_2021_enhanced.json")


if __name__ == "__main__":
    main()
