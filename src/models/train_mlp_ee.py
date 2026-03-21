from __future__ import annotations

import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.models.mlp_utils_ee import (
    build_mlp_pipeline_ee,
    build_prediction_export_ee,
    compute_regression_metrics_ee,
    load_temporal_dataset_ee,
    postprocess_predictions_ee,
    row_sum_summary_ee,
    save_json_ee,
    save_model_ee,
    split_temporal_dataset_ee,
)
from src.utils.mlp_config_ee import (
    ACTIVATION_GRID_EE,
    ALPHA_GRID_EE,
    FEATURE_COLUMNS_EE,
    FORWARD_DATA_PATH_EE,
    HIDDEN_LAYER_SIZES_GRID_EE,
    LEARNING_RATE_INIT_GRID_EE,
    MAX_ITER_GRID_EE,
    MODEL_OUTPUT_DIR_EE,
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
    print(
        f"{split_name}: "
        f"MAE={metrics['overall_mae']:.4f}, "
        f"RMSE={metrics['overall_rmse']:.4f}, "
        f"R2={metrics['overall_r2']:.4f}"
    )
    export = build_prediction_export_ee(dataframe, raw_predictions, normalized_predictions, TARGET_COLUMNS_EE)
    return metrics, export


def _tune_hyperparameters_ee(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> tuple[dict[str, object], dict[str, float]]:
    best_params: dict[str, object] | None = None
    best_metrics: dict[str, float] | None = None

    for hidden_layer_sizes in HIDDEN_LAYER_SIZES_GRID_EE:
        for activation in ACTIVATION_GRID_EE:
            for alpha in ALPHA_GRID_EE:
                for learning_rate_init in LEARNING_RATE_INIT_GRID_EE:
                    for max_iter in MAX_ITER_GRID_EE:
                        pipeline = build_mlp_pipeline_ee(
                            feature_columns=FEATURE_COLUMNS_EE,
                            hidden_layer_sizes=hidden_layer_sizes,
                            activation=activation,
                            alpha=alpha,
                            learning_rate_init=learning_rate_init,
                            max_iter=max_iter,
                        )
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=ConvergenceWarning)
                            pipeline.fit(train_df, train_df[TARGET_COLUMNS_EE])
                        metrics, _ = _evaluate_split_ee(
                            pipeline,
                            validation_df,
                            (
                                "validation "
                                f"hidden_layer_sizes={hidden_layer_sizes} "
                                f"activation={activation} "
                                f"alpha={alpha} "
                                f"learning_rate_init={learning_rate_init} "
                                f"max_iter={max_iter}"
                            ),
                        )
                        if best_metrics is None or metrics["overall_mae"] < best_metrics["overall_mae"]:
                            best_params = {
                                "hidden_layer_sizes": hidden_layer_sizes,
                                "activation": activation,
                                "alpha": float(alpha),
                                "learning_rate_init": float(learning_rate_init),
                                "max_iter": int(max_iter),
                            }
                            best_metrics = metrics

    if best_params is None or best_metrics is None:
        raise RuntimeError("Hyperparameter tuning failed to produce a valid MLP model.")

    return best_params, best_metrics


def main() -> None:
    MODEL_OUTPUT_DIR_EE.mkdir(parents=True, exist_ok=True)

    training_frame = load_temporal_dataset_ee(TRAIN_DATA_PATH_EE)
    forward_frame = load_temporal_dataset_ee(FORWARD_DATA_PATH_EE)
    train_df, validation_df, test_df = split_temporal_dataset_ee(training_frame)

    best_params, validation_metrics = _tune_hyperparameters_ee(train_df, validation_df)

    train_validation_df = pd.concat([train_df, validation_df], ignore_index=True)
    final_pipeline = build_mlp_pipeline_ee(
        feature_columns=FEATURE_COLUMNS_EE,
        hidden_layer_sizes=best_params["hidden_layer_sizes"],
        activation=best_params["activation"],
        alpha=best_params["alpha"],
        learning_rate_init=best_params["learning_rate_init"],
        max_iter=best_params["max_iter"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        final_pipeline.fit(train_validation_df, train_validation_df[TARGET_COLUMNS_EE])

    test_metrics, test_export = _evaluate_split_ee(final_pipeline, test_df, "test")
    forward_metrics, forward_export = _evaluate_split_ee(final_pipeline, forward_frame, "external_forward")

    save_model_ee(final_pipeline, MODEL_OUTPUT_DIR_EE / "mlp_model_ee.joblib")
    save_json_ee(
        {
            "best_params_ee": {
                **best_params,
                "hidden_layer_sizes": list(best_params["hidden_layer_sizes"]),
            },
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
        MODEL_OUTPUT_DIR_EE / "mlp_metrics_ee.json",
    )
    test_export.to_csv(MODEL_OUTPUT_DIR_EE / "mlp_test_predictions_ee.csv", index=False)
    forward_export.to_csv(MODEL_OUTPUT_DIR_EE / "mlp_external_predictions_ee.csv", index=False)

    print("\nMLP EE summary")
    print(
        "Best params: "
        f"hidden_layer_sizes={best_params['hidden_layer_sizes']}, "
        f"activation={best_params['activation']}, "
        f"alpha={best_params['alpha']}, "
        f"learning_rate_init={best_params['learning_rate_init']}, "
        f"max_iter={best_params['max_iter']}"
    )
    print(
        "Test overall metrics: "
        f"MAE={test_metrics['overall_mae']:.4f}, "
        f"RMSE={test_metrics['overall_rmse']:.4f}, "
        f"R2={test_metrics['overall_r2']:.4f}"
    )
    print(
        "Forward overall metrics: "
        f"MAE={forward_metrics['overall_mae']:.4f}, "
        f"RMSE={forward_metrics['overall_rmse']:.4f}, "
        f"R2={forward_metrics['overall_r2']:.4f}"
    )
    print(f"Artifacts written to: {MODEL_OUTPUT_DIR_EE}")


if __name__ == "__main__":
    main()
