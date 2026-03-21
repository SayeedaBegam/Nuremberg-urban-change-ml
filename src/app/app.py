from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import streamlit as st
from shapely.geometry import shape
from streamlit_folium import st_folium

# Make the repository root importable when Streamlit runs the file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.explain_utils import helpful_explanation, limitations_text, misleading_explanation
from src.app.map_utils import build_map
from src.app.viz_utils import (
    CLASS_ORDER,
    average_composition,
    boxplot_chart,
    category_metrics_frame,
    change_summary_long,
    composition_stacked_bar,
    composition_summary_long,
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
from src.utils.config import INTERIM_DIR, MODELS_DIR, PROCESSED_DIR


APP_PREDICTIONS_PATH = PROCESSED_DIR / "app_predictions.csv"
METRICS_PATH = MODELS_DIR / "metrics.json"
GRID_PATH = INTERIM_DIR / "grid.geojson"
PROCESSED_EE_DIR = PROJECT_ROOT / "data" / "processed_esa_ee"
PREDICTION_GRID_PATHS = [
    PROCESSED_EE_DIR / "nuremberg_2020_composition_250m.csv",
    PROCESSED_EE_DIR / "nuremberg_2021_composition_250m.csv",
    PROCESSED_EE_DIR / "nuremberg_2019_features_250m.csv",
]
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


def _load_grid() -> gpd.GeoDataFrame:
    if not GRID_PATH.exists():
        _stop_with_file_error(GRID_PATH, "grid file")

    grid = gpd.read_file(GRID_PATH)
    if "cell_id" not in grid.columns:
        st.error(f"Grid file is missing required column `cell_id`: `{GRID_PATH}`")
        st.stop()
    grid["cell_id"] = grid["cell_id"].astype(str)
    return grid.to_crs(4326)


def _load_prediction_grid() -> gpd.GeoDataFrame | None:
    for path in PREDICTION_GRID_PATHS:
        if not path.exists():
            continue

        frame = pd.read_csv(path, usecols=["cell_id", ".geo"])
        if frame.empty:
            continue

        geometry = frame[".geo"].map(lambda value: shape(json.loads(value)))
        gdf = gpd.GeoDataFrame(frame[["cell_id"]].copy(), geometry=geometry, crs=3857)
        gdf["cell_id"] = gdf["cell_id"].astype(str)
        return gdf.to_crs(4326)
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


def _load_predictions() -> pd.DataFrame:
    if not APP_PREDICTIONS_PATH.exists():
        _stop_with_file_error(
            APP_PREDICTIONS_PATH,
            "app predictions export",
            "source .venv/bin/activate && python -m src.models.export_app_artifacts",
        )

    predictions = pd.read_csv(APP_PREDICTIONS_PATH)
    predictions = _normalize_legacy_predictions(predictions)
    required_columns = {"cell_id", "model", "split"}
    missing = required_columns - set(predictions.columns)
    if missing:
        st.error(f"`{APP_PREDICTIONS_PATH}` is missing required columns: {sorted(missing)}")
        st.stop()

    predictions["cell_id"] = predictions["cell_id"].astype(str)
    predictions = _compute_delta_columns(predictions)
    return prepare_dashboard_frame(predictions)


def _load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    with METRICS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    with col2:
        predicted_positive = positive_change_share(frame, "pred_delta_{class_name}")
        actual_positive = positive_change_share(frame, "actual_delta_{class_name}")
        pie_cols = st.columns(2)
        with pie_cols[0]:
            if not predicted_positive.empty and predicted_positive["value"].sum() > 0:
                st.altair_chart(pie_chart(predicted_positive, "Predicted positive change"), use_container_width=True)
            else:
                st.info("No positive predicted change is present for the current selection.")
        with pie_cols[1]:
            if not actual_positive.empty and actual_positive["value"].sum() > 0:
                st.altair_chart(pie_chart(actual_positive, "Actual positive change"), use_container_width=True)


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
        st.info("Category-wise metrics are unavailable in `metrics.json`.")
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
        st.info("Overall metrics are unavailable in `metrics.json`.")
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


def _render_downloads(filtered_frame: pd.DataFrame, comparison_frame: pd.DataFrame) -> None:
    st.divider()
    st.header("Download / Export")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download filtered prediction table",
            data=filtered_frame.to_csv(index=False).encode("utf-8"),
            file_name="filtered_predictions.csv",
            mime="text/csv",
        )
    with col2:
        if comparison_frame.empty:
            st.info("Model comparison summary is unavailable for download.")
        else:
            st.download_button(
                label="Download model comparison summary",
                data=comparison_frame.to_csv(index=False).encode("utf-8"),
                file_name="model_comparison_summary.csv",
                mime="text/csv",
            )


predictions = _load_predictions()
metrics = _load_metrics()

grid = _load_grid()
prediction_grid = _load_prediction_grid()
if prediction_grid is not None:
    grid_ids = set(grid["cell_id"])
    prediction_ids = set(predictions["cell_id"].astype(str))
    if grid_ids.isdisjoint(prediction_ids):
        grid = prediction_grid

available_models = sorted(predictions["model"].dropna().unique().tolist())
available_splits = sorted(predictions["split"].dropna().unique().tolist())

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
    st.error(f"Missing value column `{predicted_column}` in `{APP_PREDICTIONS_PATH}`.")
    st.stop()

merged = grid.merge(filtered_predictions, on="cell_id", how="left")
predicted_tooltip_columns = _build_tooltip_columns(merged, predicted_column, actual_column)
legend_name = f"{selected_model} | {selected_split} | {selected_layer_mode} | {selected_class}"
show_actual_composition = (
    selected_layer_mode == "composition"
    and actual_column is not None
    and actual_column in merged.columns
    and merged[actual_column].notna().any()
)

# Keep the existing top visible layout intact.
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
        st.write(helpful_explanation(selected_layer_mode, selected_class))
        st.subheader("Potentially Misleading Explanation")
        st.write(misleading_explanation())
        st.subheader("Important Limitations")
        st.write(limitations_text())
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
        st.write(helpful_explanation(selected_layer_mode, selected_class))
        st.subheader("Potentially Misleading Explanation")
        st.write(misleading_explanation())
        st.subheader("Important Limitations")
        st.write(limitations_text())

# New vertically stacked analysis sections below the current top layout.
_render_composition_summary(filtered_predictions)
_render_uncertainty_section(filtered_predictions, merged)
_render_change_summary(filtered_predictions)
_render_error_analysis(filtered_predictions)
comparison_summary_frame = _render_model_comparison(metrics)
_render_dominant_summary(filtered_predictions)
_render_downloads(filtered_predictions, comparison_summary_frame)
