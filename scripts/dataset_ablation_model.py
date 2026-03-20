"""
Dataset Ablation Study: Train models with different data sources

This script trains Random Forest models using different combinations of
satellite datasets and compares:
1. Prediction accuracy (MAE, RMSE)
2. Model confidence (uncertainty estimates)
3. Feature importance by data source
4. Performance improvement % per dataset combination
"""

from __future__ import annotations

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from src.models.evaluate import regression_metrics
from src.models.uncertainty import random_forest_uncertainty
from src.utils.config import CHANGE_TARGET_COLUMNS, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE


class DatasetAblationStudy:
    """Run ablation study comparing different satellite data sources."""

    def __init__(self):
        self.results = {}
        self.baseline_metrics = None

    def _spatial_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Approximate spatial split using x-centroid."""
        if "centroid_x_t1" in df.columns:
            cutoff = df["centroid_x_t1"].median()
            return df[df["centroid_x_t1"] <= cutoff], df[df["centroid_x_t1"] > cutoff]
        midpoint = len(df) // 2
        ordered = df.sort_values("cell_id").reset_index(drop=True)
        return ordered.iloc[:midpoint], ordered.iloc[midpoint:]

    def _identify_feature_groups(self, df: pd.DataFrame) -> dict[str, list[str]]:
        """Identify which source each feature came from."""
        all_columns = set(df.columns)
        excluded = {
            "cell_id", "year_label_t1", "year_label_t2", "change_binary",
            "centroid_x_t1", "centroid_y_t1", "centroid_x_t2", "centroid_y_t2",
            *CHANGE_TARGET_COLUMNS,
            "built_up_prop_t1", "vegetation_prop_t1", "water_prop_t1", "other_prop_t1",
            "built_up_prop_t2", "vegetation_prop_t2", "water_prop_t2", "other_prop_t2",
        }

        features = [c for c in all_columns if c not in excluded]

        # Feature groups based on naming
        groups = {
            "raw_bands": [f for f in features if any(b in f for b in ["b2_", "b3_", "b4_", "b8_", "b11_", "b12_"])],
            "spectral_indices": [f for f in features if any(idx in f for idx in ["ndvi_", "ndbi_", "ndwi_", "brightness_"])],
        }

        return groups

    def _filter_features_by_group(
        self,
        all_features: list[str],
        groups_to_include: list[str],
        available_groups: dict[str, list[str]]
    ) -> list[str]:
        """Filter to specific feature groups."""
        selected = []
        for group in groups_to_include:
            if group in available_groups:
                selected.extend(available_groups[group])
        return selected

    def _train_model(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
        feature_names: list[str]
    ) -> tuple[dict, np.ndarray, float]:
        """Train RF model and return metrics, uncertainty, training time."""
        if not feature_names:
            return {}, np.zeros(len(y_test)), 0.0

        x_train_filtered = x_train[feature_names]
        x_test_filtered = x_test[feature_names]

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=14,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
                n_jobs=1,
            )),
        ])

        model.fit(x_train_filtered, y_train)
        y_pred = pd.DataFrame(
            model.predict(x_test_filtered),
            columns=CHANGE_TARGET_COLUMNS,
            index=y_test.index
        )

        metrics = regression_metrics(y_test, y_pred)

        # Extract uncertainty for built_up target
        rf_model = model.named_steps["model"]
        imputer = model.named_steps["imputer"]
        x_test_imputed = imputer.transform(x_test_filtered)
        built_up_rf = rf_model.estimators_[0]
        uncertainty = random_forest_uncertainty(built_up_rf, x_test_imputed)

        return metrics, uncertainty, float(len(feature_names))

    def run(self) -> None:
        """Execute the dataset ablation study."""
        print("\n" + "="*70)
        print("DATASET ABLATION STUDY: Feature Source Comparison")
        print("="*70)

        # Load data
        print("\nLoading dataset...")
        dataset = pd.read_csv(PROCESSED_DIR / "change_dataset.csv")
        train_df, test_df = self._spatial_split(dataset)

        all_features = self._identify_feature_groups(dataset)
        full_feature_list = []
        for group_features in all_features.values():
            full_feature_list.extend(group_features)

        print(f"Dataset: {len(train_df)} train, {len(test_df)} test")
        print(f"Feature groups identified: {list(all_features.keys())}")
        print(f"Total features: {len(full_feature_list)}\n")

        x_train = train_df[full_feature_list]
        x_test = test_df[full_feature_list]
        y_train = train_df[CHANGE_TARGET_COLUMNS]
        y_test = test_df[CHANGE_TARGET_COLUMNS]

        # Baseline: All features
        print("Training baseline model (all features)...")
        baseline_metrics, baseline_unc, n_features = self._train_model(
            x_train, x_test, y_train, y_test, full_feature_list
        )

        self.results["baseline_all_features"] = {
            "metrics": baseline_metrics,
            "n_features": n_features,
            "avg_uncertainty": float(baseline_unc.mean()),
            "description": "All available features (baseline)",
        }

        self.baseline_metrics = baseline_metrics

        # Experiment 1: Raw bands only
        print("\nTraining model with raw bands only...")
        raw_bands_metrics, raw_bands_unc, n_rb = self._train_model(
            x_train, x_test, y_train, y_test, all_features["raw_bands"]
        )
        self.results["raw_bands_only"] = {
            "metrics": raw_bands_metrics,
            "n_features": n_rb,
            "avg_uncertainty": float(raw_bands_unc.mean()),
            "description": "Raw spectral bands (B2-B4, B8, B11, B12)",
        }

        # Experiment 2: Spectral indices only
        print("Training model with spectral indices only...")
        spec_idx_metrics, spec_idx_unc, n_si = self._train_model(
            x_train, x_test, y_train, y_test, all_features["spectral_indices"]
        )
        self.results["spectral_indices_only"] = {
            "metrics": spec_idx_metrics,
            "n_features": n_si,
            "avg_uncertainty": float(spec_idx_unc.mean()),
            "description": "Derived indices (NDVI, NDBI, NDWI, brightness)",
        }

        self._print_results()
        self._save_results()

    def _print_results(self) -> None:
        """Print ablation results to console."""
        print("\n" + "="*70)
        print("DATASET ABLATION RESULTS")
        print("="*70)

        for exp_name, exp_data in self.results.items():
            print(f"\n{exp_name.upper()}")
            print(f"  Description: {exp_data['description']}")
            print(f"  # Features: {exp_data['n_features']}")

            if exp_data["metrics"]:
                avg_mae = np.mean([v for k, v in exp_data["metrics"].items() if "_mae" in k])
                avg_rmse = np.mean([v for k, v in exp_data["metrics"].items() if "_rmse" in k])
                avg_unc = exp_data["avg_uncertainty"]

                print(f"  Avg MAE:  {avg_mae:.6f}")
                print(f"  Avg RMSE: {avg_rmse:.6f}")
                print(f"  Avg Uncertainty: {avg_unc:.6f}")

                # Compare to baseline
                if exp_name != "baseline_all_features":
                    baseline_mae = np.mean([v for k, v in self.baseline_metrics.items() if "_mae" in k])
                    mae_change = ((avg_mae - baseline_mae) / baseline_mae) * 100
                    print(f"  MAE vs Baseline: {mae_change:+.2f}%")

    def _save_results(self) -> None:
        """Save results to JSON."""
        output_file = MODELS_DIR / "dataset_ablation_results.json"

        # Prepare JSON-serializable output
        results_for_json = {}
        for exp_name, exp_data in self.results.items():
            results_for_json[exp_name] = {
                "description": exp_data["description"],
                "n_features": exp_data["n_features"],
                "avg_uncertainty": exp_data["avg_uncertainty"],
                "metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                           for k, v in exp_data["metrics"].items()},
            }

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results_for_json, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")


def main() -> None:
    """Run the dataset ablation study."""
    study = DatasetAblationStudy()
    study.run()


if __name__ == "__main__":
    main()
