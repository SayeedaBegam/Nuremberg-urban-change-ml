from __future__ import annotations

import pandas as pd

from src.models.common_utils_ee_osm import (
    build_prediction_export_ee_osm,
    compute_regression_metrics_ee_osm,
    get_final_targets_ee_osm,
    get_training_targets_ee_osm,
    load_temporal_dataset_ee_osm,
    postprocess_three_target_predictions_ee_osm,
    row_sum_summary_ee_osm,
    save_json_ee_osm,
    save_model_ee_osm,
    split_temporal_dataset_ee_osm,
    three_target_sum_summary_ee_osm,
)
from src.models.elastic_net_utils_ee_osm import build_elastic_net_pipeline_ee_osm, export_coefficients_ee_osm
from src.utils.elastic_net_config_ee_osm import (
    ALPHA_GRID_EE_OSM,
    FEATURE_COLUMNS_EE_OSM,
    FORWARD_DATA_PATH_EE_OSM,
    L1_RATIO_GRID_EE_OSM,
    MODEL_OUTPUT_DIR_EE_OSM,
    TRAIN_DATA_PATH_EE_OSM,
    TRAIN_TARGET_COLUMNS_EE_OSM,
)

MODEL_NAME_EE_OSM = "elastic_net"


def _evaluate_split_ee_osm(pipeline, dataframe: pd.DataFrame, split_name: str) -> tuple[dict[str, float], pd.DataFrame]:
    raw_predictions = pipeline.predict(dataframe)
    raw_three_frame, final_predictions = postprocess_three_target_predictions_ee_osm(raw_predictions)
    metrics = compute_regression_metrics_ee_osm(
        y_true=get_final_targets_ee_osm(dataframe),
        y_pred=final_predictions,
    )
    metrics["n_rows"] = int(len(dataframe))
    metrics["prediction_sum3_before_ee_osm"] = three_target_sum_summary_ee_osm(raw_three_frame)
    metrics["prediction_row_sums_after_ee_osm"] = row_sum_summary_ee_osm(final_predictions)
    print(
        f"{split_name}: "
        f"MAE={metrics['overall_mae']:.4f}, "
        f"RMSE={metrics['overall_rmse']:.4f}, "
        f"R2={metrics['overall_r2']:.4f}"
    )
    export = build_prediction_export_ee_osm(
        dataframe,
        raw_three_frame,
        final_predictions,
        model_name=MODEL_NAME_EE_OSM,
        split_name=split_name,
    )
    return metrics, export


def _tune_hyperparameters_ee_osm(train_df: pd.DataFrame, validation_df: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    best_params: dict[str, float] | None = None
    best_metrics: dict[str, float] | None = None

    for alpha in ALPHA_GRID_EE_OSM:
        for l1_ratio in L1_RATIO_GRID_EE_OSM:
            pipeline = build_elastic_net_pipeline_ee_osm(
                feature_columns=FEATURE_COLUMNS_EE_OSM,
                alpha=alpha,
                l1_ratio=l1_ratio,
            )
            pipeline.fit(train_df, get_training_targets_ee_osm(train_df))
            metrics, _ = _evaluate_split_ee_osm(
                pipeline,
                validation_df,
                f"validation alpha={alpha} l1_ratio={l1_ratio}",
            )
            if best_metrics is None or metrics["overall_mae"] < best_metrics["overall_mae"]:
                best_params = {"alpha": float(alpha), "l1_ratio": float(l1_ratio)}
                best_metrics = metrics

    if best_params is None or best_metrics is None:
        raise RuntimeError("Hyperparameter tuning failed to produce a valid Elastic Net model.")
    return best_params, best_metrics


def main() -> None:
    MODEL_OUTPUT_DIR_EE_OSM.mkdir(parents=True, exist_ok=True)

    training_frame = load_temporal_dataset_ee_osm(TRAIN_DATA_PATH_EE_OSM)
    forward_frame = load_temporal_dataset_ee_osm(FORWARD_DATA_PATH_EE_OSM)
    train_df, validation_df, test_df = split_temporal_dataset_ee_osm(training_frame)

    best_params, validation_metrics = _tune_hyperparameters_ee_osm(train_df, validation_df)

    train_validation_df = pd.concat([train_df, validation_df], ignore_index=True)
    final_pipeline = build_elastic_net_pipeline_ee_osm(
        feature_columns=FEATURE_COLUMNS_EE_OSM,
        alpha=best_params["alpha"],
        l1_ratio=best_params["l1_ratio"],
    )
    final_pipeline.fit(train_validation_df, get_training_targets_ee_osm(train_validation_df))

    test_metrics, test_export = _evaluate_split_ee_osm(final_pipeline, test_df, "test")
    forward_metrics, forward_export = _evaluate_split_ee_osm(final_pipeline, forward_frame, "external_forward")
    coefficient_frame = export_coefficients_ee_osm(final_pipeline, FEATURE_COLUMNS_EE_OSM, TRAIN_TARGET_COLUMNS_EE_OSM)

    save_model_ee_osm(final_pipeline, MODEL_OUTPUT_DIR_EE_OSM / "elastic_net_model_ee_osm.joblib")
    save_json_ee_osm(
        {
            "best_params_ee_osm": best_params,
            "feature_columns_ee_osm": FEATURE_COLUMNS_EE_OSM,
            "trained_target_columns_ee_osm": TRAIN_TARGET_COLUMNS_EE_OSM,
            "final_target_columns_ee_osm": ["vegetation", "built_up", "water", "other"],
            "row_counts_ee_osm": {
                "train": int(len(train_df)),
                "validation": int(len(validation_df)),
                "test": int(len(test_df)),
                "external_forward": int(len(forward_frame)),
            },
            "validation_metrics_ee_osm": validation_metrics,
            "test_metrics_ee_osm": test_metrics,
            "external_forward_metrics_ee_osm": forward_metrics,
        },
        MODEL_OUTPUT_DIR_EE_OSM / "elastic_net_metrics_ee_osm.json",
    )
    coefficient_frame.to_csv(MODEL_OUTPUT_DIR_EE_OSM / "elastic_net_coefficients_ee_osm.csv", index=False)
    test_export.to_csv(MODEL_OUTPUT_DIR_EE_OSM / "elastic_net_test_predictions_ee_osm.csv", index=False)
    forward_export.to_csv(MODEL_OUTPUT_DIR_EE_OSM / "elastic_net_external_predictions_ee_osm.csv", index=False)

    print("\nElastic Net EE OSM summary")
    print(f"Best params: alpha={best_params['alpha']}, l1_ratio={best_params['l1_ratio']}")
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
    print(f"Artifacts written to: {MODEL_OUTPUT_DIR_EE_OSM}")


if __name__ == "__main__":
    main()
