from __future__ import annotations

import pandas as pd

from src.models.xgboost_utils_ee import (
    build_prediction_export_ee,
    build_xgboost_pipeline_ee,
    compute_regression_metrics_ee,
    export_feature_importances_ee,
    get_xgboost_version_ee,
    is_xgboost_available_ee,
    load_temporal_dataset_ee,
    postprocess_predictions_ee,
    row_sum_summary_ee,
    save_json_ee,
    save_model_ee,
    split_temporal_dataset_ee,
)
from src.utils.xgboost_config_ee import (
    COLSAMPLE_BYTREE_GRID_EE,
    FEATURE_COLUMNS_EE,
    FORWARD_DATA_PATH_EE,
    LEARNING_RATE_GRID_EE,
    MAX_DEPTH_GRID_EE,
    MODEL_OUTPUT_DIR_EE,
    N_ESTIMATORS_GRID_EE,
    REG_ALPHA_GRID_EE,
    REG_LAMBDA_GRID_EE,
    SUBSAMPLE_GRID_EE,
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
) -> tuple[dict[str, int | float], dict[str, float]]:
    best_params: dict[str, int | float] | None = None
    best_metrics: dict[str, float] | None = None

    for n_estimators in N_ESTIMATORS_GRID_EE:
        for max_depth in MAX_DEPTH_GRID_EE:
            for learning_rate in LEARNING_RATE_GRID_EE:
                for subsample in SUBSAMPLE_GRID_EE:
                    for colsample_bytree in COLSAMPLE_BYTREE_GRID_EE:
                        for reg_alpha in REG_ALPHA_GRID_EE:
                            for reg_lambda in REG_LAMBDA_GRID_EE:
                                pipeline = build_xgboost_pipeline_ee(
                                    feature_columns=FEATURE_COLUMNS_EE,
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    learning_rate=learning_rate,
                                    subsample=subsample,
                                    colsample_bytree=colsample_bytree,
                                    reg_alpha=reg_alpha,
                                    reg_lambda=reg_lambda,
                                )
                                pipeline.fit(train_df, train_df[TARGET_COLUMNS_EE])
                                metrics, _ = _evaluate_split_ee(
                                    pipeline,
                                    validation_df,
                                    (
                                        "validation "
                                        f"n_estimators={n_estimators} "
                                        f"max_depth={max_depth} "
                                        f"learning_rate={learning_rate} "
                                        f"subsample={subsample} "
                                        f"colsample_bytree={colsample_bytree} "
                                        f"reg_alpha={reg_alpha} "
                                        f"reg_lambda={reg_lambda}"
                                    ),
                                )
                                if best_metrics is None or metrics["overall_mae"] < best_metrics["overall_mae"]:
                                    best_params = {
                                        "n_estimators": int(n_estimators),
                                        "max_depth": int(max_depth),
                                        "learning_rate": float(learning_rate),
                                        "subsample": float(subsample),
                                        "colsample_bytree": float(colsample_bytree),
                                        "reg_alpha": float(reg_alpha),
                                        "reg_lambda": float(reg_lambda),
                                    }
                                    best_metrics = metrics

    if best_params is None or best_metrics is None:
        raise RuntimeError("Hyperparameter tuning failed to produce a valid XGBoost model.")

    return best_params, best_metrics


def main() -> None:
    if not is_xgboost_available_ee():
        raise ModuleNotFoundError(
            "xgboost is not installed in the current environment. "
            "Install it manually, for example with `pip install xgboost`, before running `python -m src.models.train_xgboost_ee`."
        )

    print(f"Using xgboost version: {get_xgboost_version_ee()}")
    MODEL_OUTPUT_DIR_EE.mkdir(parents=True, exist_ok=True)

    training_frame = load_temporal_dataset_ee(TRAIN_DATA_PATH_EE)
    forward_frame = load_temporal_dataset_ee(FORWARD_DATA_PATH_EE)
    train_df, validation_df, test_df = split_temporal_dataset_ee(training_frame)

    best_params, validation_metrics = _tune_hyperparameters_ee(train_df, validation_df)

    train_validation_df = pd.concat([train_df, validation_df], ignore_index=True)
    final_pipeline = build_xgboost_pipeline_ee(
        feature_columns=FEATURE_COLUMNS_EE,
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        reg_alpha=best_params["reg_alpha"],
        reg_lambda=best_params["reg_lambda"],
    )
    final_pipeline.fit(train_validation_df, train_validation_df[TARGET_COLUMNS_EE])

    test_metrics, test_export = _evaluate_split_ee(final_pipeline, test_df, "test")
    forward_metrics, forward_export = _evaluate_split_ee(final_pipeline, forward_frame, "external_forward")
    feature_importances = export_feature_importances_ee(final_pipeline, FEATURE_COLUMNS_EE, TARGET_COLUMNS_EE)

    save_model_ee(final_pipeline, MODEL_OUTPUT_DIR_EE / "xgboost_model_ee.joblib")
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
            "xgboost_version_ee": get_xgboost_version_ee(),
        },
        MODEL_OUTPUT_DIR_EE / "xgboost_metrics_ee.json",
    )
    feature_importances.to_csv(MODEL_OUTPUT_DIR_EE / "xgboost_feature_importances_ee.csv", index=False)
    test_export.to_csv(MODEL_OUTPUT_DIR_EE / "xgboost_test_predictions_ee.csv", index=False)
    forward_export.to_csv(MODEL_OUTPUT_DIR_EE / "xgboost_external_predictions_ee.csv", index=False)

    print("\nXGBoost EE summary")
    print(
        "Best params: "
        f"n_estimators={best_params['n_estimators']}, "
        f"max_depth={best_params['max_depth']}, "
        f"learning_rate={best_params['learning_rate']}, "
        f"subsample={best_params['subsample']}, "
        f"colsample_bytree={best_params['colsample_bytree']}, "
        f"reg_alpha={best_params['reg_alpha']}, "
        f"reg_lambda={best_params['reg_lambda']}"
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
