"""
Ablation Study: Feature Importance for Urban Change Prediction

This script systematically removes feature groups and measures the impact on
model performance. We use the Random Forest model trained on the full feature set
as a baseline, then measure performance degradation when key feature groups
are removed.

Feature groups:
- Raw bands: band statistics (mean/std for B2, B3, B4, B8, B11, B12)
- Vegetation indices: NDVI (mean, std)
- Moisture index: NDWI
- Built-up index: NDBI
- Brightness: mean brightness across visible bands
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from src.models.evaluate import regression_metrics
from src.utils.config import CHANGE_TARGET_COLUMNS, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE


def _spatial_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Approximate a spatial split using x-centroid median."""
    if "centroid_x_t1" in df.columns:
        cutoff = df["centroid_x_t1"].median()
        return df[df["centroid_x_t1"] <= cutoff], df[df["centroid_x_t1"] > cutoff]
    midpoint = len(df) // 2
    ordered = df.sort_values("cell_id").reset_index(drop=True)
    return ordered.iloc[:midpoint], ordered.iloc[midpoint:]


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Get all feature columns (excluding targets, metadata, IDs)."""
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
        "built_up_prop_t1",
        "vegetation_prop_t1",
        "water_prop_t1",
        "other_prop_t1",
        "built_up_prop_t2",
        "vegetation_prop_t2",
        "water_prop_t2",
        "other_prop_t2",
    }
    return [col for col in df.columns if col not in excluded]


def _define_feature_groups() -> dict[str, list[str]]:
    """Define feature groups for ablation study.

    Returns dict: {group_name: [list of matching patterns]}
    Patterns are matched as substring in feature names.
    """
    return {
        "raw_bands": ["b2_", "b3_", "b4_", "b8_", "b11_", "b12_"],
        "ndvi": ["ndvi_"],
        "ndbi": ["ndbi_"],
        "ndwi": ["ndwi_"],
        "brightness": ["brightness_"],
    }


def _filter_features(all_features: list[str], include_patterns: list[str]) -> list[str]:
    """Return features matching any of the include_patterns."""
    return [f for f in all_features if any(p in f for p in include_patterns)]


def _train_ablated_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, float]:
    """Train RF model with specified features and return metrics."""
    if not feature_names:
        return {f"{target}_mae": np.nan for target in CHANGE_TARGET_COLUMNS for _ in ["mae"]}

    x_train_filtered = x_train[feature_names]
    x_test_filtered = x_test[feature_names]

    model = Pipeline(
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

    model.fit(x_train_filtered, y_train)
    y_pred = pd.DataFrame(
        model.predict(x_test_filtered),
        columns=CHANGE_TARGET_COLUMNS,
        index=y_test.index
    )

    return regression_metrics(y_test, y_pred)


def main() -> None:
    """Run the ablation study."""
    print("Loading dataset...")
    dataset = pd.read_csv(PROCESSED_DIR / "change_dataset.csv")
    train_df, test_df = _spatial_split(dataset)

    all_features = _feature_columns(dataset)
    x_train = train_df[all_features]
    x_test = test_df[all_features]
    y_train = train_df[CHANGE_TARGET_COLUMNS]
    y_test = test_df[CHANGE_TARGET_COLUMNS]

    print(f"Dataset loaded: {len(train_df)} train, {len(test_df)} test")
    print(f"Features: {len(all_features)}")
    print(f"Targets: {CHANGE_TARGET_COLUMNS}\n")

    # Baseline: all features
    print("Training baseline model (all features)...")
    baseline_metrics = _train_ablated_model(x_train, x_test, y_train, y_test, all_features)

    # Ablation experiments
    feature_groups = _define_feature_groups()
    ablation_results = {"baseline": baseline_metrics}

    for group_name, patterns in feature_groups.items():
        print(f"\nAblating feature group: {group_name}")
        features_to_use = [f for f in all_features if not any(p in f for p in patterns)]
        print(f"  Features remaining: {len(features_to_use)} / {len(all_features)}")

        metrics = _train_ablated_model(x_train, x_test, y_train, y_test, features_to_use)
        ablation_results[f"without_{group_name}"] = metrics

    # Compute performance degradation
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)

    results_table = []

    for experiment_name, metrics in ablation_results.items():
        row = {"experiment": experiment_name}

        # Average MAE across all targets
        mae_values = [v for k, v in metrics.items() if "_mae" in k]
        rmse_values = [v for k, v in metrics.items() if "_rmse" in k]

        row["avg_mae"] = np.nanmean(mae_values) if mae_values else np.nan
        row["avg_rmse"] = np.nanmean(rmse_values) if rmse_values else np.nan

        # Per-target metrics
        for target in CHANGE_TARGET_COLUMNS:
            row[f"{target}_mae"] = metrics.get(f"{target}_mae", np.nan)
            row[f"{target}_rmse"] = metrics.get(f"{target}_rmse", np.nan)

        results_table.append(row)

    results_df = pd.DataFrame(results_table)

    # Compute degradation relative to baseline
    baseline_mae = results_df[results_df["experiment"] == "baseline"]["avg_mae"].values[0]
    baseline_rmse = results_df[results_df["experiment"] == "baseline"]["avg_rmse"].values[0]

    results_df["mae_increase_%"] = (
        (results_df["avg_mae"] - baseline_mae) / baseline_mae * 100
    )
    results_df["rmse_increase_%"] = (
        (results_df["avg_rmse"] - baseline_rmse) / baseline_rmse * 100
    )

    # Display results
    print("\nBaseline (all features):")
    print(f"  Average MAE:  {baseline_mae:.6f}")
    print(f"  Average RMSE: {baseline_rmse:.6f}")

    print("\nFeature Group Importance (impact of removing each group):")
    print("-" * 70)

    for _, row in results_df[results_df["experiment"] != "baseline"].iterrows():
        print(f"\n{row['experiment']}:")
        print(f"  MAE:  {row['avg_mae']:.6f} (↑ {row['mae_increase_%']:+.1f}%)")
        print(f"  RMSE: {row['avg_rmse']:.6f} (↑ {row['rmse_increase_%']:+.1f}%)")

        # Per-target details
        for target in CHANGE_TARGET_COLUMNS:
            mae_val = row[f"{target}_mae"]
            print(f"    {target:<20} MAE: {mae_val:.6f}")

    # Save results to CSV
    output_file = MODELS_DIR / "ablation_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Detailed results saved to: {output_file}")

    # Summary ranking
    print("\n" + "="*70)
    print("FEATURE GROUP IMPORTANCE RANKING (by MAE degradation)")
    print("="*70)

    ablation_only = results_df[results_df["experiment"] != "baseline"].copy()
    ablation_only = ablation_only.sort_values("mae_increase_%", ascending=False)

    for idx, (_, row) in enumerate(ablation_only.iterrows(), 1):
        print(f"{idx}. {row['experiment']:<25} {row['mae_increase_%']:+6.1f}% worse")

    print("\nInterpretation:")
    print("- Higher % = more important (bigger impact when removed)")
    print("- Negative % = slightly better (overfitting mitigation)")


if __name__ == "__main__":
    main()
