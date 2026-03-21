from __future__ import annotations

import pandas as pd

from src.models.random_forest_utils_ee import (
    build_prediction_export_ee,
    build_random_forest_pipeline_ee,
    compute_regression_metrics_ee,
    estimate_uncertainty_ee,
    export_feature_importances_ee,
    load_temporal_dataset_ee,
    postprocess_predictions_ee,
    row_sum_summary_ee,
    save_json_ee,
    save_model_ee,
    split_temporal_dataset_ee,
)
from src.utils.random_forest_config_ee import (
    FEATURE_COLUMNS_EE,
    FORWARD_DATA_PATH_EE,
    MAX_DEPTH_GRID_EE,
    MAX_FEATURES_GRID_EE,
    MIN_SAMPLES_LEAF_GRID_EE,
    MODEL_OUTPUT_DIR_EE,
    N_ESTIMATORS_GRID_EE,
    TARGET_COLUMNS_EE,
    TRAIN_DATA_PATH_EE,
)


def _evaluate_split_ee(
    pipeline,
    dataframe: pd.DataFrame,
    split_name: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    raw_predictions = pipeline.predict(dataframe)
    normalized_predictions = postprocess_predictions_ee(raw_predictions, TARGET_COLUMNS_EE)
    uncertainty_frame = estimate_uncertainty_ee(pipeline, dataframe, TARGET_COLUMNS_EE)
    metrics = compute_regression_metrics_ee(
        y_true=dataframe[TARGET_COLUMNS_EE],
        y_pred=normalized_predictions,
        target_columns=TARGET_COLUMNS_EE,
    )
    metrics["n_rows"] = int(len(dataframe))
    metrics["prediction_row_sums_before_ee"] = row_sum_summary_ee(
        pd.DataFrame(raw_predictions, columns=TARGET_COLUMNS_EE)
    )
    metrics["prediction_row_sums_after_ee"] = row_sum_summary_ee(normalized_predictions)
    metrics["uncertainty_mean_row"] = float(uncertainty_frame["uncertainty_mean_row"].mean())
    print(
        f"{split_name}: "
        f"MAE={metrics['overall_mae']:.4f}, "
        f"RMSE={metrics['overall_rmse']:.4f}, "
        f"R2={metrics['overall_r2']:.4f}, "
        f"mean_uncertainty={metrics['uncertainty_mean_row']:.4f}"
    )
    export = build_prediction_export_ee(
        dataframe,
        raw_predictions,
        normalized_predictions,
        uncertainty_frame,
        TARGET_COLUMNS_EE,
    )
    return metrics, export


def _tune_hyperparameters_ee(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> tuple[dict[str, int | float | str | None], dict[str, float]]:
    best_params: dict[str, int | float | str | None] | None = None
    best_metrics: dict[str, float] | None = None

    for n_estimators in N_ESTIMATORS_GRID_EE:
        for max_depth in MAX_DEPTH_GRID_EE:
            for min_samples_leaf in MIN_SAMPLES_LEAF_GRID_EE:
                for max_features in MAX_FEATURES_GRID_EE:
                    pipeline = build_random_forest_pipeline_ee(
                        feature_columns=FEATURE_COLUMNS_EE,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                    )
                    pipeline.fit(train_df, train_df[TARGET_COLUMNS_EE])
                    metrics, _ = _evaluate_split_ee(
                        pipeline,
                        validation_df,
                        (
                            "validation "
                            f"n_estimators={n_estimators} "
                            f"max_depth={max_depth} "
                            f"min_samples_leaf={min_samples_leaf} "
                            f"max_features={max_features}"
                        ),
                    )
                    if best_metrics is None or metrics["overall_mae"] < best_metrics["overall_mae"]:
                        best_params = {
                            "n_estimators": int(n_estimators),
                            "max_depth": None if max_depth is None else int(max_depth),
                            "min_samples_leaf": int(min_samples_leaf),
                            "max_features": max_features,
                        }
                        best_metrics = metrics

    if best_params is None or best_metrics is None:
        raise RuntimeError("Hyperparameter tuning failed to produce a valid Random Forest model.")

    return best_params, best_metrics


def main() -> None:
    MODEL_OUTPUT_DIR_EE.mkdir(parents=True, exist_ok=True)

    training_frame = load_temporal_dataset_ee(TRAIN_DATA_PATH_EE)
    forward_frame = load_temporal_dataset_ee(FORWARD_DATA_PATH_EE)
    train_df, validation_df, test_df = split_temporal_dataset_ee(training_frame)

    best_params, validation_metrics = _tune_hyperparameters_ee(train_df, validation_df)

    train_validation_df = pd.concat([train_df, validation_df], ignore_index=True)
    final_pipeline = build_random_forest_pipeline_ee(
        feature_columns=FEATURE_COLUMNS_EE,
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        max_features=best_params["max_features"],
    )
    final_pipeline.fit(train_validation_df, train_validation_df[TARGET_COLUMNS_EE])

    test_metrics, test_export = _evaluate_split_ee(final_pipeline, test_df, "test")
    forward_metrics, forward_export = _evaluate_split_ee(final_pipeline, forward_frame, "external_forward")
    feature_importances = export_feature_importances_ee(final_pipeline, FEATURE_COLUMNS_EE, TARGET_COLUMNS_EE)

    save_model_ee(final_pipeline, MODEL_OUTPUT_DIR_EE / "random_forest_model_ee.joblib")
    save_json_ee(
        {
            "best_params_ee": best_params,
            "feature_columns_ee": FEATURE_COLUMNS_EE,
            "target_columns_ee": TARGET_COLUMNS_EE,
            "row_counts_ee": {
                "train": int(len(train_df)),
                "validation": int(len(validation_df)),
                "test": int(len(test_df)),
                "external_forward": int(len(forward_frame)),
            },
            "validation_metrics_ee": validation_metrics,
            "test_metrics_ee": test_metrics,
            "external_forward_metrics_ee": forward_metrics,
        },
        MODEL_OUTPUT_DIR_EE / "random_forest_metrics_ee.json",
    )
    feature_importances.to_csv(MODEL_OUTPUT_DIR_EE / "random_forest_feature_importances_ee.csv", index=False)
    test_export.to_csv(MODEL_OUTPUT_DIR_EE / "random_forest_test_predictions_ee.csv", index=False)
    forward_export.to_csv(MODEL_OUTPUT_DIR_EE / "random_forest_external_predictions_ee.csv", index=False)

    print("\nRandom Forest EE summary")
    print(
        "Best params: "
        f"n_estimators={best_params['n_estimators']}, "
        f"max_depth={best_params['max_depth']}, "
        f"min_samples_leaf={best_params['min_samples_leaf']}, "
        f"max_features={best_params['max_features']}"
    )
    print(
        "Test overall metrics: "
        f"MAE={test_metrics['overall_mae']:.4f}, "
        f"RMSE={test_metrics['overall_rmse']:.4f}, "
        f"R2={test_metrics['overall_r2']:.4f}, "
        f"mean_uncertainty={test_metrics['uncertainty_mean_row']:.4f}"
    )
    print(
        "Forward overall metrics: "
        f"MAE={forward_metrics['overall_mae']:.4f}, "
        f"RMSE={forward_metrics['overall_rmse']:.4f}, "
        f"R2={forward_metrics['overall_r2']:.4f}, "
        f"mean_uncertainty={forward_metrics['uncertainty_mean_row']:.4f}"
    )
    print(f"Artifacts written to: {MODEL_OUTPUT_DIR_EE}")


if __name__ == "__main__":
    main()
