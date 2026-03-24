from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import streamlit as st
from shapely.geometry import shape
from streamlit_folium import st_folium

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.explain_utils import helpful_explanation, limitations_text, misleading_explanation
from src.app.map_utils import build_map
from src.app.pipeline_registry import PIPELINE_LABELS, PIPELINE_LABEL_TO_KEY, PIPELINE_REGISTRY
from src.app.stress_utils import feature_columns_for_pipeline, run_stress_test
from src.app.viz_utils import (
    CLASS_ORDER,
    average_composition,
    boxplot_chart,
    category_metrics_frame,
    change_summary_long,
    coefficient_bar_chart,
    composition_stacked_bar,
    composition_summary_long,
    confusion_matrix_chart,
    confusion_text_overlay,
    dominant_summary,
    error_scatter_frame,
    grouped_bar_chart,
    histogram_chart,
    overall_metrics_frame,
    pie_chart,
    positive_change_share,
    prepare_dashboard_frame,
    scatter_chart,
    top_rows,
    uncertainty_long,
)


LAND_COVER_CLASSES = CLASS_ORDER
LAYER_MODES = ["composition", "change"]


st.set_page_config(page_title="Nuremberg Urban Change", layout="wide")
st.title("Mapping Urban Change in Nuremberg")
st.caption("Tabular machine learning on Sentinel-2-derived features and ESA WorldCover land-cover proportions.")


def _stop_with_file_error(path: Path, description: str, command: str | None = None) -> None:
    st.error(f"Missing {description}: `{path}`")
    if command:
        st.code(command, language="bash")
    st.stop()


def _load_geojson_grid(path: Path) -> gpd.GeoDataFrame | None:
    if not path.exists():
        return None
    grid = gpd.read_file(path)
    if "cell_id" not in grid.columns:
        return None
    grid["cell_id"] = grid["cell_id"].astype(str)
    return grid.to_crs(4326)


def _load_csv_geometry(path: Path) -> gpd.GeoDataFrame | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, usecols=["cell_id", ".geo"])
    except ValueError:
        return None
    if frame.empty:
        return None
    geometry = frame[".geo"].dropna().map(lambda value: shape(json.loads(value)))
    valid = frame.loc[geometry.index, ["cell_id"]].copy()
    gdf = gpd.GeoDataFrame(valid, geometry=geometry, crs=3857)
    gdf["cell_id"] = gdf["cell_id"].astype(str)
    return gdf.to_crs(4326)


def _load_grid_candidate(path: Path) -> gpd.GeoDataFrame | None:
    if path.suffix.lower() == ".geojson":
        return _load_geojson_grid(path)
    if path.suffix.lower() == ".csv":
        return _load_csv_geometry(path)
    return None


def _load_primary_grid(pipeline_config: dict) -> gpd.GeoDataFrame:
    grid_path = pipeline_config["grid_path"]
    grid = _load_grid_candidate(grid_path)
    if grid is not None:
        return grid

    fallback_grid = _load_prediction_grid(pipeline_config)
    if fallback_grid is not None:
        st.warning(f"Primary grid is unavailable for this data source. Using fallback geometry from `{fallback_grid.attrs.get('source_path', 'pipeline artifacts')}`.")
        return fallback_grid

    _stop_with_file_error(grid_path, "grid file")


def _load_prediction_grid(pipeline_config: dict) -> gpd.GeoDataFrame | None:
    for path in pipeline_config.get("grid_fallback_paths", []):
        grid = _load_grid_candidate(path)
        if grid is not None:
            grid.attrs["source_path"] = str(path)
            return grid
    return None


def _normalize_legacy_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "model" not in normalized.columns:
        normalized["model"] = "random_forest"
    if "split" not in normalized.columns:
        normalized["split"] = "legacy"
    return normalized


def _compute_delta_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    for class_name in LAND_COVER_CLASSES:
        pred_t2 = f"pred_{class_name}_prop_t2"
        actual_t2 = f"actual_{class_name}_prop_t2"
        t1 = f"{class_name}_prop_t1"
        if pred_t2 in enriched.columns and t1 in enriched.columns:
            enriched[f"pred_delta_{class_name}"] = enriched[pred_t2] - enriched[t1]
        if actual_t2 in enriched.columns and t1 in enriched.columns:
            enriched[f"actual_delta_{class_name}"] = enriched[actual_t2] - enriched[t1]
    return enriched


def _load_t1_composition(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    composition = pd.read_csv(path)
    required = ["cell_id", "built_up", "vegetation", "water", "other"]
    missing = [column for column in required if column not in composition.columns]
    if missing:
        return None
    rename_map = {class_name: f"{class_name}_prop_t1" for class_name in LAND_COVER_CLASSES}
    result = composition[required].rename(columns=rename_map).copy()
    result["cell_id"] = result["cell_id"].astype(str)
    return result


def _normalize_ee_predictions(predictions_path: Path) -> pd.DataFrame:
    if not predictions_path.exists():
        _stop_with_file_error(
            predictions_path,
            "app predictions export",
            "source .venv/bin/activate && python -m src.models.export_app_artifacts",
        )
    predictions = pd.read_csv(predictions_path)
    predictions = _normalize_legacy_predictions(predictions)
    required_columns = {"cell_id", "model", "split"}
    missing = required_columns - set(predictions.columns)
    if missing:
        st.error(f"`{predictions_path}` is missing required columns: {sorted(missing)}")
        st.stop()
    predictions["cell_id"] = predictions["cell_id"].astype(str)
    predictions["pipeline"] = "ee"
    return predictions


def _normalize_ee_osm_prediction_export(
    path: Path,
    model_name: str,
    split_name: str,
    pipeline_key: str,
    t1_composition_path: Path | None,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rename_map = {}
    for class_name in LAND_COVER_CLASSES:
        actual_column = f"actual_{class_name}"
        pred_column = f"pred_{class_name}"
        if actual_column in frame.columns:
            rename_map[actual_column] = f"actual_{class_name}_prop_t2"
        if pred_column in frame.columns:
            rename_map[pred_column] = f"pred_{class_name}_prop_t2"
    normalized = frame.rename(columns=rename_map).copy()
    normalized["cell_id"] = normalized["cell_id"].astype(str)
    normalized["model"] = model_name
    normalized["split"] = split_name
    normalized["pipeline"] = pipeline_key

    t1_composition = _load_t1_composition(t1_composition_path)
    if t1_composition is not None:
        normalized = normalized.merge(t1_composition, on="cell_id", how="left")

    return normalized


def _load_predictions(pipeline_key: str, pipeline_config: dict) -> pd.DataFrame:
    if pipeline_key == "ee":
        predictions = _normalize_ee_predictions(pipeline_config["predictions_path"])
        return prepare_dashboard_frame(_compute_delta_columns(predictions))

    frames = []
    missing_exports = []
    for model_name, split_mapping in pipeline_config.get("prediction_exports", {}).items():
        for split_name, path in split_mapping.items():
            if not path.exists():
                missing_exports.append(str(path))
                continue
            frames.append(
                _normalize_ee_osm_prediction_export(
                    path,
                    model_name=model_name,
                    split_name=split_name,
                    pipeline_key=pipeline_key,
                    t1_composition_path=pipeline_config.get("t1_composition_by_split", {}).get(split_name),
                )
            )

    if not frames:
        st.error(
            "No prediction exports were found for the selected data source. "
            "Train the corresponding models first or check that the exported CSV files exist."
        )
        if missing_exports:
            st.code("\n".join(missing_exports), language="text")
        st.stop()

    if missing_exports:
        st.info("Some prediction exports are missing for the selected data source. Available files are shown; missing ones are skipped.")

    predictions = pd.concat(frames, ignore_index=True)
    return prepare_dashboard_frame(_compute_delta_columns(predictions))


def _normalize_ee_osm_metrics(pipeline_config: dict) -> dict:
    metrics_payload = {}
    missing_metrics = []
    for model_name, path in pipeline_config.get("metrics_exports", {}).items():
        if not path.exists():
            missing_metrics.append(str(path))
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metrics_payload[model_name] = {
            "best_params": payload.get("best_params_ee_osm", {}),
            "feature_columns": payload.get("feature_columns_ee_osm", []),
            "target_columns": payload.get("final_target_columns_ee_osm", []),
            "row_counts": payload.get("row_counts_ee_osm", {}),
            "metrics_by_split": {
                "test_2019_2020": payload.get("test_metrics_ee_osm", {}),
                "forward_2020_2021": payload.get("external_forward_metrics_ee_osm", {}),
            },
        }
    if not metrics_payload and missing_metrics:
        st.info("Metrics files are unavailable for the selected data source, so only prediction-driven views will be shown.")
    return metrics_payload


def _load_metrics(pipeline_key: str, pipeline_config: dict) -> dict:
    if pipeline_key == "ee":
        metrics_path = pipeline_config["metrics_path"]
        if not metrics_path.exists():
            return {}
        with metrics_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return _normalize_ee_osm_metrics(pipeline_config)


def _selected_columns(layer_mode: str, class_name: str) -> tuple[str, str | None]:
    if layer_mode == "composition":
        return f"pred_{class_name}_prop_t2", f"actual_{class_name}_prop_t2"
    return f"pred_delta_{class_name}", f"actual_delta_{class_name}"


def _metrics_for_selection(metrics: dict, model_name: str, split_name: str) -> tuple[dict, dict]:
    model_metrics = metrics.get(model_name, {}) if isinstance(metrics, dict) else {}
    if not isinstance(model_metrics, dict):
        return {}, {}

    best_params = model_metrics.get("best_params", {})
    split_metrics = model_metrics.get("metrics_by_split", {}).get(split_name, {})
    if not isinstance(best_params, dict):
        best_params = {}
    if not isinstance(split_metrics, dict):
        split_metrics = {}
    return best_params, split_metrics


def _display_metrics(best_params: dict, split_metrics: dict, class_name: str) -> None:
    st.sidebar.markdown("### Evaluation")
    if not split_metrics:
        st.sidebar.info("No metrics found for the selected model and split.")
        return

    class_prefix = f"{class_name}_t2"
    summary = {
        key: split_metrics[key]
        for key in [
            "n_rows",
            "overall_mae",
            "overall_rmse",
            "overall_r2",
            f"{class_prefix}_mae",
            f"{class_prefix}_rmse",
            f"{class_prefix}_r2",
            f"{class_name}_mae",
            f"{class_name}_rmse",
            f"{class_name}_r2",
            "uncertainty_mean_row",
            "uncertainty_mean",
        ]
        if key in split_metrics
    }
    if best_params:
        st.sidebar.markdown("#### Best Params")
        st.sidebar.json(best_params)
    st.sidebar.json(summary)


def _build_tooltip_columns(frame: pd.DataFrame, primary_column: str, secondary_column: str | None = None) -> list[str]:
    tooltip_columns = ["cell_id", primary_column]
    if secondary_column and secondary_column in frame.columns:
        tooltip_columns.append(secondary_column)
    if "uncertainty_built_up" in frame.columns:
        tooltip_columns.append("uncertainty_built_up")
    return tooltip_columns


def _change_mode_available(frame: pd.DataFrame, predicted_column: str) -> bool:
    return predicted_column in frame.columns and frame[predicted_column].notna().any()


def _actual_map_title(split_name: str) -> str:
    year_token = split_name.split("_")[-1]
    return f"Actual {year_token} labels" if year_token.isdigit() else "Actual T2 labels"


def _available_uncertainty_columns(frame: pd.DataFrame) -> list[str]:
    columns = []
    if "uncertainty_mean" in frame.columns and frame["uncertainty_mean"].notna().any():
        columns.append("uncertainty_mean")
    columns.extend(
        [
            f"uncertainty_{class_name}"
            for class_name in LAND_COVER_CLASSES
            if f"uncertainty_{class_name}" in frame.columns and frame[f"uncertainty_{class_name}"].notna().any()
        ]
    )
    return columns


def _actual_available(frame: pd.DataFrame) -> bool:
    return any(f"actual_{class_name}_prop_t2" in frame.columns for class_name in LAND_COVER_CLASSES)


def _render_composition_summary(frame: pd.DataFrame) -> None:
    st.divider()
    st.header("Land-Cover Composition Summary")

    summary_long = composition_summary_long(frame)
    if summary_long.empty:
        st.info("Composition summary is unavailable because the required T1/T2 composition columns are missing.")
        return

    col1, col2 = st.columns([1.35, 1.15])
    with col1:
        st.altair_chart(composition_stacked_bar(summary_long), use_container_width=True)

    with col2:
        pie_cols = st.columns(2)
        predicted_avg = average_composition(frame, "pred_{class_name}_prop_t2")
        actual_avg = average_composition(frame, "actual_{class_name}_prop_t2")
        with pie_cols[0]:
            if not predicted_avg.empty:
                st.altair_chart(pie_chart(predicted_avg, "Predicted T2 average"), use_container_width=True)
        with pie_cols[1]:
            if not actual_avg.empty:
                st.altair_chart(pie_chart(actual_avg, "Actual T2 average"), use_container_width=True)
            else:
                st.info("Actual T2 composition is unavailable for the current selection.")


def _render_uncertainty_section(frame: pd.DataFrame, merged: gpd.GeoDataFrame) -> None:
    st.divider()
    st.header("Confidence / Uncertainty Analysis")

    uncertainty_columns = _available_uncertainty_columns(frame)
    if not uncertainty_columns:
        st.info("No uncertainty columns are available in the current prediction export.")
        return

    selected_uncertainty = st.selectbox(
        "Uncertainty view",
        uncertainty_columns,
        index=uncertainty_columns.index("uncertainty_mean") if "uncertainty_mean" in uncertainty_columns else 0,
        key="uncertainty_selector",
    )
    uncertainty_melted = uncertainty_long(frame)

    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(
            histogram_chart(frame, selected_uncertainty, selected_uncertainty.replace("_", " ")),
            use_container_width=True,
        )
    with col2:
        if uncertainty_melted.empty:
            st.info("Class-wise uncertainty columns are not available for the current selection.")
        else:
            st.altair_chart(
                boxplot_chart(uncertainty_melted, "uncertainty", "class", "Uncertainty by class"),
                use_container_width=True,
            )

    table_columns = ["cell_id", selected_uncertainty, "uncertainty_mean"] + [f"uncertainty_{class_name}" for class_name in LAND_COVER_CLASSES]
    top_uncertainty = top_rows(frame, selected_uncertainty, table_columns, n_rows=12)
    if not top_uncertainty.empty:
        st.dataframe(top_uncertainty, use_container_width=True)

    if st.checkbox("Show uncertainty heat map", value=False, key="uncertainty_map_toggle"):
        uncertainty_map = build_map(
            merged,
            value_column=selected_uncertainty,
            tooltip_columns=_build_tooltip_columns(merged, selected_uncertainty),
            layer_mode="composition",
            legend_name=f"uncertainty | {selected_uncertainty}",
        )
        st_folium(uncertainty_map, width=1100, height=500)


def _render_change_summary(frame: pd.DataFrame) -> None:
    st.divider()
    st.header("Land-Cover Change Summary")

    change_long = change_summary_long(frame)
    if change_long.empty:
        st.info("Change summary is unavailable because the required T1 composition columns are missing.")
        return

    col1, col2 = st.columns([1.35, 1.15])
    with col1:
        st.altair_chart(
            grouped_bar_chart(change_long, "class", "value", "source", "Average delta"),
            use_container_width=True,
        )

    predicted_positive = positive_change_share(frame, "pred_delta_{class_name}")
    actual_positive = positive_change_share(frame, "actual_delta_{class_name}")
    predicted_has_positive = not predicted_positive.empty and predicted_positive["value"].sum() > 0
    actual_has_positive = not actual_positive.empty and actual_positive["value"].sum() > 0

    with col2:
        if not predicted_has_positive and not actual_has_positive:
            st.info(
                "Positive-change pies are unavailable for this selection because the average class deltas do not contain positive change to summarize. The grouped bar chart on the left still shows the mean signed deltas."
            )
        else:
            pie_cols = st.columns(2)
            with pie_cols[0]:
                if predicted_has_positive:
                    st.altair_chart(pie_chart(predicted_positive, "Predicted positive change"), use_container_width=True)
                else:
                    st.info("Predicted positive-change pie is unavailable because the mean predicted deltas are not positive for this selection.")
            with pie_cols[1]:
                if actual_has_positive:
                    st.altair_chart(pie_chart(actual_positive, "Actual positive change"), use_container_width=True)
                else:
                    st.info("Actual positive-change pie is unavailable because the mean actual deltas are not positive for this selection.")


def _render_error_analysis(frame: pd.DataFrame) -> None:
    st.divider()
    st.header("Prediction Error Analysis")

    if not _actual_available(frame):
        st.info("Actual T2 columns are unavailable, so prediction error analysis cannot be shown.")
        return

    error_selector = st.selectbox(
        "Error class",
        ["built_up", "vegetation", "water", "other", "mean"],
        key="error_selector",
    )
    error_column = "abs_error_mean" if error_selector == "mean" else f"abs_error_{error_selector}"
    if error_column not in frame.columns:
        st.info(f"The column `{error_column}` is unavailable for the current selection.")
        return

    col1, col2 = st.columns([1.0, 1.2])
    with col1:
        st.altair_chart(histogram_chart(frame, error_column, error_column.replace("_", " ")), use_container_width=True)
    with col2:
        if error_selector == "mean":
            st.info("Predicted vs actual scatter is shown for individual classes only. Select a specific class to view it.")
        else:
            scatter_data = error_scatter_frame(frame, error_selector)
            if scatter_data.empty:
                st.info("Predicted vs actual scatter is unavailable for the current selection.")
            else:
                st.altair_chart(scatter_chart(scatter_data, f"Predicted vs actual: {error_selector}"), use_container_width=True)

    worst_columns = [
        "cell_id",
        error_column,
        f"pred_{error_selector}_prop_t2" if error_selector != "mean" else None,
        f"actual_{error_selector}_prop_t2" if error_selector != "mean" else None,
        "abs_error_mean",
        "uncertainty_mean",
    ]
    worst_table = top_rows(frame, error_column, [column for column in worst_columns if column], n_rows=12)
    if not worst_table.empty:
        st.dataframe(worst_table, use_container_width=True)


def _render_model_comparison(metrics_payload: dict) -> pd.DataFrame:
    st.divider()
    st.header("Model Comparison Across Splits")

    summary_tables = []

    st.subheader("Category-wise Comparison")
    selected_metric_class = st.selectbox("Category", LAND_COVER_CLASSES, key="comparison_category_selector")
    category_frame = category_metrics_frame(metrics_payload, selected_metric_class)
    if category_frame.empty:
        st.info("Category-wise metrics are unavailable for the selected data source.")
    else:
        chart_cols = st.columns(3)
        for column_container, metric_name, label in zip(chart_cols, ["r2", "mae", "rmse"], ["R²", "MAE", "RMSE"]):
            metric_df = category_frame.dropna(subset=[metric_name])
            if metric_df.empty:
                column_container.info(f"No {label} values available.")
            else:
                column_container.altair_chart(
                    grouped_bar_chart(metric_df, "model", metric_name, "split", label),
                    use_container_width=True,
                )
        st.dataframe(category_frame, use_container_width=True)
        category_export = category_frame.copy()
        category_export.insert(0, "section", f"category:{selected_metric_class}")
        summary_tables.append(category_export)

    st.subheader("Overall Comparison")
    overall_frame = overall_metrics_frame(metrics_payload)
    if overall_frame.empty:
        st.info("Overall metrics are unavailable for the selected data source.")
    else:
        chart_cols = st.columns(3)
        for column_container, metric_name, label in zip(chart_cols, ["r2", "mae", "rmse"], ["Overall R²", "Overall MAE", "Overall RMSE"]):
            metric_df = overall_frame.dropna(subset=[metric_name])
            if metric_df.empty:
                column_container.info(f"No {label} values available.")
            else:
                column_container.altair_chart(
                    grouped_bar_chart(metric_df, "model", metric_name, "split", label),
                    use_container_width=True,
                )
        st.dataframe(overall_frame, use_container_width=True)
        overall_export = overall_frame.copy()
        overall_export.insert(0, "section", "overall")
        summary_tables.append(overall_export)

    return pd.concat(summary_tables, ignore_index=True) if summary_tables else pd.DataFrame()


def _render_dominant_summary(frame: pd.DataFrame) -> None:
    dominant_counts, dominant_margins = dominant_summary(frame)
    if dominant_counts.empty:
        return

    st.divider()
    st.header("Dominant Land-Cover Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(pie_chart(dominant_counts.rename(columns={"count": "value"}), "Dominant predicted class"), use_container_width=True)
    with col2:
        if dominant_margins.empty:
            st.info("Dominant margin values are unavailable.")
        else:
            st.altair_chart(
                grouped_bar_chart(dominant_margins.rename(columns={"margin_bin": "bin", "count": "value"}), "bin", "value", "bin", "Dominant margin distribution"),
                use_container_width=True,
            )


def _render_downloads(filtered_frame: pd.DataFrame, comparison_frame: pd.DataFrame, pipeline_key: str) -> None:
    st.divider()
    st.header("Download / Export")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download filtered prediction table",
            data=filtered_frame.to_csv(index=False).encode("utf-8"),
            file_name=f"filtered_predictions_{pipeline_key}.csv",
            mime="text/csv",
        )
    with col2:
        if comparison_frame.empty:
            st.info("Model comparison summary is unavailable for download.")
        else:
            st.download_button(
                label="Download model comparison summary",
                data=comparison_frame.to_csv(index=False).encode("utf-8"),
                file_name=f"model_comparison_summary_{pipeline_key}.csv",
                mime="text/csv",
            )


def _change_analysis_columns_available(frame: pd.DataFrame) -> bool:
    required_columns = []
    for class_name in LAND_COVER_CLASSES:
        required_columns.extend(
            [
                f"{class_name}_prop_t1",
                f"pred_{class_name}_prop_t2",
                f"actual_{class_name}_prop_t2",
            ]
        )
    return all(column in frame.columns for column in required_columns)


def _derived_change_frame(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    derived = frame.copy()
    actual_total = pd.Series(0.0, index=derived.index, dtype=float)
    predicted_total = pd.Series(0.0, index=derived.index, dtype=float)
    for class_name in LAND_COVER_CLASSES:
        t1_column = f"{class_name}_prop_t1"
        pred_column = f"pred_{class_name}_prop_t2"
        actual_column = f"actual_{class_name}_prop_t2"
        actual_total = actual_total + (derived[actual_column] - derived[t1_column]).abs()
        predicted_total = predicted_total + (derived[pred_column] - derived[t1_column]).abs()
    derived["actual_total_change"] = actual_total
    derived["predicted_total_change"] = predicted_total
    derived["actual_changed"] = (actual_total >= threshold).astype(int)
    derived["predicted_changed"] = (predicted_total >= threshold).astype(int)
    return derived.dropna(subset=["actual_total_change", "predicted_total_change"])


def _change_confusion_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    counts = {
        "TN": int(((frame["actual_changed"] == 0) & (frame["predicted_changed"] == 0)).sum()),
        "FP": int(((frame["actual_changed"] == 0) & (frame["predicted_changed"] == 1)).sum()),
        "FN": int(((frame["actual_changed"] == 1) & (frame["predicted_changed"] == 0)).sum()),
        "TP": int(((frame["actual_changed"] == 1) & (frame["predicted_changed"] == 1)).sum()),
    }
    matrix = pd.DataFrame(
        [
            {"actual_label": "Unchanged", "predicted_label": "Unchanged", "cell_type": "True Negative", "count": counts["TN"]},
            {"actual_label": "Unchanged", "predicted_label": "Changed", "cell_type": "False Positive", "count": counts["FP"]},
            {"actual_label": "Changed", "predicted_label": "Unchanged", "cell_type": "False Negative", "count": counts["FN"]},
            {"actual_label": "Changed", "predicted_label": "Changed", "cell_type": "True Positive", "count": counts["TP"]},
        ]
    )
    return matrix, counts


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _render_change_error_analysis(frame: pd.DataFrame, pipeline_key: str, model_name: str, split_name: str) -> None:
    st.divider()
    st.header("Change Error Analysis")

    if not _change_analysis_columns_available(frame):
        st.info(
            "Derived change analysis is unavailable for the current selection because the required T1 or actual T2 composition columns are missing."
        )
        return

    threshold = st.slider(
        "Derived change threshold",
        min_value=0.01,
        max_value=0.50,
        value=0.10,
        step=0.01,
        key=f"change_threshold_{pipeline_key}_{model_name}_{split_name}",
    )
    st.caption(
        "This section derives changed vs unchanged from regression outputs using total absolute composition change. It is not a native classifier confusion matrix."
    )

    derived = _derived_change_frame(frame, threshold)
    if derived.empty:
        st.info("No rows are available for derived change analysis after applying the required column checks.")
        return

    confusion_frame, counts = _change_confusion_frame(derived)
    tp, tn, fp, fn = counts["TP"], counts["TN"], counts["FP"], counts["FN"]
    total = tp + tn + fp + fn
    metrics_summary = {
        "False positive rate": _safe_ratio(fp, fp + tn),
        "False negative rate": _safe_ratio(fn, fn + tp),
        "Precision": _safe_ratio(tp, tp + fp),
        "Recall": _safe_ratio(tp, tp + fn),
        "F1 score": _safe_ratio(2 * tp, (2 * tp) + fp + fn),
        "Accuracy": _safe_ratio(tp + tn, total),
    }

    col1, col2 = st.columns([1.2, 1.0])
    with col1:
        st.altair_chart(confusion_matrix_chart(confusion_frame) + confusion_text_overlay(confusion_frame), use_container_width=True)
    with col2:
        metric_cols = st.columns(2)
        for idx, (label, value) in enumerate(metrics_summary.items()):
            metric_cols[idx % 2].metric(label, f"{value:.3f}")

    count_cols = st.columns(4)
    count_cols[0].metric("TP", f"{tp:,}")
    count_cols[1].metric("TN", f"{tn:,}")
    count_cols[2].metric("FP", f"{fp:,}")
    count_cols[3].metric("FN", f"{fn:,}")

    st.dataframe(
        pd.DataFrame([counts]).rename(columns={"TP": "TP", "TN": "TN", "FP": "FP", "FN": "FN"}),
        use_container_width=True,
    )


def _load_elastic_net_coefficients(pipeline_config: dict) -> pd.DataFrame:
    path = pipeline_config.get("elastic_net_coefficients_path")
    if path is None or not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required = {"target", "feature", "coefficient", "abs_coefficient"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    normalized = frame.copy()
    normalized["target_label"] = normalized["target"].astype(str).str.replace("_t2", "", regex=False)
    return normalized


def _render_elastic_net_interpretability(selected_model: str, pipeline_config: dict) -> None:
    st.divider()
    st.header("Elastic Net Interpretability")

    if selected_model != "elastic_net":
        st.caption("Showing the Elastic Net baseline for interpretability while another model is selected above.")

    coefficients = _load_elastic_net_coefficients(pipeline_config)
    if coefficients.empty:
        st.info("Elastic Net coefficient artifacts are unavailable for the selected data source.")
        return

    available_targets = [class_name for class_name in LAND_COVER_CLASSES if class_name in set(coefficients["target_label"])]
    if not available_targets:
        st.info("No readable Elastic Net targets were found in the saved coefficient artifact.")
        return

    selected_target = st.selectbox("Interpretability target", available_targets, key=f"elastic_net_target_{pipeline_config['label']}")
    target_frame = coefficients[coefficients["target_label"] == selected_target].copy()
    if target_frame.empty:
        st.info("No coefficients are available for the selected target.")
        return

    top_positive = target_frame[target_frame["coefficient"] > 0].nlargest(6, "coefficient")
    top_negative = target_frame[target_frame["coefficient"] < 0].nsmallest(6, "coefficient")
    chart_frame = pd.concat([top_negative, top_positive], ignore_index=True)
    if chart_frame.empty:
        chart_frame = target_frame.nlargest(12, "abs_coefficient").sort_values("coefficient")
    chart_frame = chart_frame.reset_index(drop=True)
    chart_frame["sort_order"] = range(len(chart_frame))

    col1, col2 = st.columns([1.1, 1.0])
    with col1:
        st.altair_chart(
            coefficient_bar_chart(chart_frame, f"Top Elastic Net coefficients: {selected_target}"),
            use_container_width=True,
        )
    with col2:
        strongest_positive = target_frame.loc[target_frame["coefficient"].idxmax()]
        strongest_negative = target_frame.loc[target_frame["coefficient"].idxmin()]
        st.markdown("**Model summary**")
        st.write(
            f"Within the fitted linear model, `{strongest_positive['feature']}` is most associated with higher predicted `{selected_target}` proportion, "
            f"while `{strongest_negative['feature']}` is most associated with lower predicted `{selected_target}` proportion. "
            "These are conditional associations within the scaled engineered feature set, not causal effects."
        )
        st.dataframe(
            target_frame[["feature", "coefficient", "abs_coefficient"]]
            .sort_values("abs_coefficient", ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
        )


def _render_stress_test_section(
    pipeline_key: str,
    pipeline_config: dict,
    model_name: str,
    split_name: str,
    selected_class: str,
    grid: gpd.GeoDataFrame,
) -> None:
    st.divider()
    st.header("Stress Test: Feature Noise Robustness")
    st.write(
        "This stress test evaluates how sensitive the model is to input perturbations by injecting noise into the features. Larger prediction shifts indicate lower robustness. This does not reflect real deployment noise but provides a controlled robustness check."
    )

    model_path = pipeline_config.get("model_artifact_paths", {}).get(model_name)
    dataset_paths = pipeline_config.get("dataset_paths_by_split", {})
    train_data_path = dataset_paths.get("test_2019_2020")
    forward_data_path = dataset_paths.get("forward_2020_2021")
    if model_path is None or train_data_path is None or forward_data_path is None:
        st.info("Stress test artifacts are not fully configured for the selected data source.")
        return
    if not model_path.exists():
        st.warning(f"Stress test model artifact is missing: `{model_path}`")
        return

    available_features = feature_columns_for_pipeline(pipeline_key)
    control_cols = st.columns([1.0, 1.0, 1.0, 1.0])
    with control_cols[0]:
        noise_level = st.slider(
            "Noise level",
            min_value=0.0,
            max_value=0.30,
            value=0.10,
            step=0.01,
            key=f"stress_noise_level_{pipeline_key}_{model_name}_{split_name}",
        )
    with control_cols[1]:
        noise_mode = st.selectbox(
            "Noise mode",
            ["All features", "Single feature"],
            index=0,
            key=f"stress_noise_mode_{pipeline_key}_{model_name}_{split_name}",
        )
    with control_cols[2]:
        stress_class = st.selectbox(
            "Stress map class",
            LAND_COVER_CLASSES,
            index=LAND_COVER_CLASSES.index(selected_class),
            key=f"stress_class_{pipeline_key}_{model_name}_{split_name}",
        )
    selected_feature = None
    if noise_mode == "Single feature":
        with control_cols[3]:
            selected_feature = st.selectbox(
                "Feature",
                available_features,
                key=f"stress_feature_{pipeline_key}_{model_name}_{split_name}",
            )

    try:
        stress_payload = run_stress_test(
            pipeline_key=pipeline_key,
            model_path=model_path,
            train_data_path=train_data_path,
            forward_data_path=forward_data_path,
            split_name=split_name,
            noise_level=noise_level,
            noise_mode=noise_mode,
            selected_feature=selected_feature,
        )
    except Exception as exc:
        st.warning(f"Stress test could not be computed for the selected model: {exc}")
        return

    result_frame = stress_payload["result_frame"]
    metrics_before = stress_payload["metrics_before"]
    metrics_after = stress_payload["metrics_after"]

    metric_table = pd.DataFrame(
        [
            {"metric": "MAE", "before": metrics_before["mae"], "after": metrics_after["mae"], "delta": metrics_after["mae"] - metrics_before["mae"]},
            {"metric": "RMSE", "before": metrics_before["rmse"], "after": metrics_after["rmse"], "delta": metrics_after["rmse"] - metrics_before["rmse"]},
            {"metric": "R²", "before": metrics_before["r2"], "after": metrics_after["r2"], "delta": metrics_after["r2"] - metrics_before["r2"]},
        ]
    )

    metric_cols = st.columns(3)
    metric_cols[0].metric("MAE after noise", f"{metrics_after['mae']:.3f}", delta=f"{metrics_after['mae'] - metrics_before['mae']:+.3f}")
    metric_cols[1].metric("RMSE after noise", f"{metrics_after['rmse']:.3f}", delta=f"{metrics_after['rmse'] - metrics_before['rmse']:+.3f}")
    metric_cols[2].metric("R² after noise", f"{metrics_after['r2']:.3f}", delta=f"{metrics_after['r2'] - metrics_before['r2']:+.3f}")
    st.dataframe(metric_table, use_container_width=True)

    summary_cols = st.columns(2)
    summary_cols[0].metric("Mean absolute prediction shift", f"{stress_payload['mean_shift']:.4f}")
    summary_cols[1].metric("Max shift", f"{stress_payload['max_shift']:.4f}")

    col1, col2 = st.columns([1.0, 1.2])
    with col1:
        st.altair_chart(
            histogram_chart(result_frame, "prediction_shift_mean", "|pred_noisy - pred_original|"),
            use_container_width=True,
        )
    with col2:
        shift_column = f"prediction_shift_{stress_class}"
        stress_map_frame = grid.merge(result_frame, on="cell_id", how="left")
        stress_map = build_map(
            stress_map_frame,
            value_column=shift_column,
            tooltip_columns=["cell_id", shift_column, f"pred_{stress_class}_prop_t2", f"pred_noisy_{stress_class}_prop_t2"],
            layer_mode="composition",
            legend_name=f"stress shift | {model_name} | {split_name} | {stress_class}",
        )
        st_folium(stress_map, width=900, height=420)



selected_pipeline_label = st.sidebar.selectbox("Data source", PIPELINE_LABELS, index=0)
selected_pipeline = PIPELINE_LABEL_TO_KEY[selected_pipeline_label]
pipeline_config = PIPELINE_REGISTRY[selected_pipeline]
if pipeline_config.get("sidebar_note"):
    st.sidebar.caption(pipeline_config["sidebar_note"])

predictions = _load_predictions(selected_pipeline, pipeline_config)
metrics = _load_metrics(selected_pipeline, pipeline_config)

grid = _load_primary_grid(pipeline_config)
prediction_grid = _load_prediction_grid(pipeline_config)
if prediction_grid is not None:
    grid_ids = set(grid["cell_id"])
    prediction_ids = set(predictions["cell_id"].astype(str))
    if grid_ids.isdisjoint(prediction_ids):
        grid = prediction_grid
        st.info("Using pipeline-specific geometry because the primary grid IDs do not overlap with the selected prediction export.")

loaded_models = set(predictions["model"].dropna().astype(str))
available_models = [model for model in pipeline_config["supported_models"] if model in loaded_models]
if not available_models:
    available_models = sorted(loaded_models)

loaded_splits = set(predictions["split"].dropna().astype(str))
available_splits = [split for split in pipeline_config["supported_splits"] if split in loaded_splits]
if not available_splits:
    available_splits = sorted(loaded_splits)

selected_model = st.sidebar.selectbox("Model", available_models)
selected_split = st.sidebar.selectbox("Split", available_splits)
selected_layer_mode = st.sidebar.selectbox("Layer mode", LAYER_MODES)
selected_class = st.sidebar.selectbox("Class", LAND_COVER_CLASSES)

st.sidebar.markdown("### Limits")
st.sidebar.write(
    "Use this dashboard for broad pattern exploration only. It is not suitable for parcel-level or policy decisions."
)

best_params, split_metrics = _metrics_for_selection(metrics, selected_model, selected_split)
_display_metrics(best_params, split_metrics, selected_class)

filtered_predictions = predictions[
    (predictions["model"] == selected_model) & (predictions["split"] == selected_split)
].copy()
if filtered_predictions.empty:
    st.error("No rows matched the selected model and split.")
    st.stop()

predicted_column, actual_column = _selected_columns(selected_layer_mode, selected_class)
if selected_layer_mode == "change" and not _change_mode_available(filtered_predictions, predicted_column):
    st.warning(
        "Change mode is unavailable for this split because the required T1 composition columns are not present in the "
        "exported artifacts. Composition mode is still available."
    )
    st.stop()

if predicted_column not in filtered_predictions.columns:
    st.error(f"Missing value column `{predicted_column}` for the selected data source.")
    st.stop()

merged = grid.merge(filtered_predictions, on="cell_id", how="left")
predicted_tooltip_columns = _build_tooltip_columns(merged, predicted_column, actual_column)
legend_name = f"{selected_pipeline_label} | {selected_model} | {selected_split} | {selected_layer_mode} | {selected_class}"
show_actual_composition = (
    selected_layer_mode == "composition"
    and actual_column is not None
    and actual_column in merged.columns
    and merged[actual_column].notna().any()
)

if show_actual_composition:
    col1, col2, col3 = st.columns([1.35, 1.35, 1.0])

    with col1:
        st.subheader("Predicted T2")
        predicted_map = build_map(
            merged,
            value_column=predicted_column,
            tooltip_columns=predicted_tooltip_columns,
            layer_mode=selected_layer_mode,
            legend_name=legend_name,
        )
        st_folium(predicted_map, width=650, height=600)

    with col2:
        st.subheader(_actual_map_title(selected_split))
        actual_map = build_map(
            merged,
            value_column=actual_column,
            tooltip_columns=_build_tooltip_columns(merged, actual_column),
            layer_mode=selected_layer_mode,
            legend_name=f"actual | {selected_split} | composition | {selected_class}",
        )
        st_folium(actual_map, width=650, height=600)

    with col3:
        st.subheader("Explanation")
        st.write(helpful_explanation(selected_layer_mode, selected_class, selected_pipeline))
        st.subheader("Potentially Misleading Explanation")
        st.write(misleading_explanation(selected_pipeline))
        st.subheader("Important Limitations")
        st.write(limitations_text(selected_pipeline))
else:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Map")
        map_obj = build_map(
            merged,
            value_column=predicted_column,
            tooltip_columns=predicted_tooltip_columns,
            layer_mode=selected_layer_mode,
            legend_name=legend_name,
        )
        st_folium(map_obj, width=900, height=600)

    with col2:
        st.subheader("Explanation")
        st.write(helpful_explanation(selected_layer_mode, selected_class, selected_pipeline))
        st.subheader("Potentially Misleading Explanation")
        st.write(misleading_explanation(selected_pipeline))
        st.subheader("Important Limitations")
        st.write(limitations_text(selected_pipeline))

_render_composition_summary(filtered_predictions)
_render_uncertainty_section(filtered_predictions, merged)
_render_change_summary(filtered_predictions)
_render_error_analysis(filtered_predictions)
comparison_summary_frame = _render_model_comparison(metrics)
_render_dominant_summary(filtered_predictions)
_render_downloads(filtered_predictions, comparison_summary_frame, selected_pipeline)
_render_change_error_analysis(filtered_predictions, selected_pipeline, selected_model, selected_split)
_render_elastic_net_interpretability(selected_model, pipeline_config)
_render_stress_test_section(selected_pipeline, pipeline_config, selected_model, selected_split, selected_class, grid)
