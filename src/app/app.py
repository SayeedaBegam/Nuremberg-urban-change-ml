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
    # Constants
    CLASS_ORDER,
    # Data processing
    average_composition,
    category_metrics_frame,
    change_summary_long,
    composition_summary_long,
    dominant_summary,
    error_scatter_frame,
    overall_metrics_frame,
    positive_change_share,
    prepare_dashboard_frame,
    top_rows,
    uncertainty_long,
    # Altair charts
    boxplot_chart,
    composition_stacked_bar,
    grouped_bar_chart,
    histogram_chart,
    pie_chart,
    scatter_chart,
    # Matplotlib charts
    plot_predictions_vs_truth,
    plot_residuals,
    plot_distribution_comparison,
    plot_error_distribution,
    plot_quantile_quantile,
    compute_error_metrics,
)
from src.utils.config import INTERIM_DIR, MODELS_DIR, PROCESSED_DIR


APP_PREDICTIONS_PATH = PROCESSED_DIR / "app_predictions_combined.csv"
METRICS_PATH = MODELS_DIR / "metrics.json"
GRID_PATH = INTERIM_DIR / "grid.geojson"
PROCESSED_EE_DIR = PROJECT_ROOT / "data" / "processed_esa_ee"
PREDICTION_GRID_PATHS = [
    PROCESSED_EE_DIR / "nuremberg_2020_composition_250m.csv",
    PROCESSED_EE_DIR / "nuremberg_2021_composition_250m.csv",
    PROCESSED_EE_DIR / "nuremberg_2019_features_250m.csv",
]
LAND_COVER_CLASSES = CLASS_ORDER  # Use from viz_utils
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
        _generate_grid()

    grid = gpd.read_file(GRID_PATH)
    if "cell_id" not in grid.columns:
        st.error(f"Grid file is missing required column `cell_id`: `{GRID_PATH}`")
        st.stop()
    grid["cell_id"] = grid["cell_id"].astype(str)
    return grid.to_crs(4326)


def _generate_grid() -> None:
    """Generate grid.geojson from processed dataset if it doesn't exist."""
    from shapely.geometry import box

    GRID_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load processed dataset
    dataset_path = PROCESSED_DIR / "dataset_2020.csv"
    if not dataset_path.exists():
        _stop_with_file_error(dataset_path, "dataset to generate grid from")

    dataset = pd.read_csv(dataset_path)
    grid_size = 250  # meters (GRID_SIZE_METERS)

    grid_cells = []
    for idx, row in dataset.iterrows():
        cx = row["centroid_x"]
        cy = row["centroid_y"]
        half_size = grid_size / 2
        cell = box(cx - half_size, cy - half_size, cx + half_size, cy + half_size)
        grid_cells.append({"cell_id": row["cell_id"], "geometry": cell})

    # Data is in Web Mercator (EPSG:3857), not UTM
    gdf = gpd.GeoDataFrame(grid_cells, crs="EPSG:3857")
    gdf.to_file(GRID_PATH, driver="GeoJSON")
    st.info(f"Generated grid with {len(gdf)} cells")


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
        normalized["split"] = "test_2020_2021"
    return normalized


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


def _selected_columns(layer_mode: str, class_name: str) -> tuple[str, str | None]:
    if layer_mode == "composition":
        return f"pred_{class_name}_prop_t2", f"actual_{class_name}_prop_t2"
    return f"pred_delta_{class_name}", f"actual_delta_{class_name}"


def _get_available_columns(frame: pd.DataFrame, class_name: str) -> tuple[str | None, str | None]:
    """Get actual column names from dataframe, handling legacy formats."""
    # Try predicted column formats
    pred_col = None
    for col_format in [
        f"pred_delta_{class_name}_random_forest",
        f"pred_delta_{class_name}_rf",
        f"pred_delta_{class_name}",
    ]:
        if col_format in frame.columns:
            pred_col = col_format
            break

    # Actual column is more standard
    actual_col = f"actual_delta_{class_name}" if f"actual_delta_{class_name}" in frame.columns else None

    return pred_col, actual_col


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

# Single model: Enhanced RandomForest (2020→2021 prediction)
selected_model = available_models[0] if available_models else "random_forest"
selected_split = available_splits[0] if available_splits else "full_2020_2021_enhanced"

st.sidebar.markdown(f"### Model: {selected_model}")
st.sidebar.info("🎯 **Enhanced RandomForest** (R² = 0.968)\n\n2020 spectral features → 2021 change prediction")
selected_layer_mode = st.sidebar.selectbox(
    "Layer mode",
    LAYER_MODES,
    index=1 if "change" in LAYER_MODES else 0  # Default to change mode
)
selected_class = st.sidebar.selectbox("Class", LAND_COVER_CLASSES)

st.sidebar.markdown("### Limits")
st.sidebar.write(
    "Use this dashboard for broad pattern exploration only. It is not suitable for parcel-level or policy decisions."
)

best_params, split_metrics = _metrics_for_selection(metrics, selected_model, selected_split)

# Only display metrics if they exist
if split_metrics:
    _display_metrics(best_params, split_metrics, selected_class)

filtered_predictions = predictions[
    (predictions["model"] == selected_model) & (predictions["split"] == selected_split)
].copy()
if filtered_predictions.empty:
    st.error("No rows matched the selected model and split.")
    st.stop()

if selected_layer_mode == "composition":
    predicted_column = f"pred_{selected_class}_prop_t2"
    actual_column = f"actual_{selected_class}_prop_t2"
else:
    # For change mode, use the helper function to find available columns
    predicted_column, actual_column = _get_available_columns(filtered_predictions, selected_class)
    if predicted_column is None:
        st.error(f"No predicted delta columns found for {selected_class}.")
        st.stop()

if predicted_column not in filtered_predictions.columns:
    st.error(f"Missing value column `{predicted_column}` in `{APP_PREDICTIONS_PATH}`.")
    st.stop()

merged = grid.merge(filtered_predictions, on="cell_id", how="left")
predicted_tooltip_columns = _build_tooltip_columns(merged, predicted_column, actual_column)
legend_name = f"{selected_model} | {selected_split} | {selected_layer_mode} | {selected_class}"

# Show map and explanation
# For change mode with actual data, show side-by-side comparison; otherwise show single map
show_comparison = (selected_layer_mode == "change" and
                   actual_column and
                   actual_column in filtered_predictions.columns and
                   filtered_predictions[actual_column].notna().any())

if show_comparison:
    # Calculate bounds to ensure both maps show the same area
    plot_merged = merged.copy()
    plot_merged[predicted_column] = pd.to_numeric(plot_merged[predicted_column], errors="coerce")
    minx, miny, maxx, maxy = plot_merged.total_bounds
    map_bounds = [[miny, minx], [maxy, maxx]]

    # Create tabs for uncertainty
    tab1, tab2 = st.tabs(["📍 Change Maps", "📊 Prediction Uncertainty"])

    with tab1:
        col1, col2, col3 = st.columns([2.2, 2.2, 0.8])

        with col1:
            st.subheader("Predicted Change")
            predicted_map = build_map(
                merged,
                value_column=predicted_column,
                tooltip_columns=predicted_tooltip_columns,
                layer_mode=selected_layer_mode,
                legend_name=f"Predicted {selected_class} change",
            )
            predicted_map.fit_bounds(map_bounds)
            st_folium(predicted_map, width=700, height=650)

        with col2:
            st.subheader("Actual Change")
            actual_tooltip_columns = _build_tooltip_columns(merged, actual_column, predicted_column)
            actual_map = build_map(
                merged,
                value_column=actual_column,
                tooltip_columns=actual_tooltip_columns,
                layer_mode=selected_layer_mode,
                legend_name=f"Actual {selected_class} change",
            )
            actual_map.fit_bounds(map_bounds)
            st_folium(actual_map, width=700, height=650)

        with col3:
            st.subheader("Explanation")
            st.write(helpful_explanation(selected_layer_mode, selected_class))
            st.subheader("Potentially Misleading Explanation")
            st.write(misleading_explanation())
            st.subheader("Important Limitations")
            st.write(limitations_text())

    with tab2:
        st.subheader("Model Prediction Uncertainty")
        if "uncertainty_built_up" in merged.columns:
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Uncertainty Map** - Shows where model is most confident (light) vs uncertain (dark)")
                uncertainty_map = build_map(
                    merged,
                    value_column="uncertainty_built_up",
                    tooltip_columns=["cell_id", "uncertainty_built_up", predicted_column],
                    layer_mode="composition",  # Use composition colormap for uncertainty
                    legend_name="Prediction Uncertainty",
                )
                uncertainty_map.fit_bounds(map_bounds)
                st_folium(uncertainty_map, width=700, height=650)

            with col2:
                st.write("**Uncertainty Statistics**")
                unc_stats = merged["uncertainty_built_up"].describe()
                st.metric("Mean Uncertainty", f"{unc_stats['mean']:.4f}")
                st.metric("Std Uncertainty", f"{unc_stats['std']:.4f}")
                st.metric("Min Uncertainty", f"{unc_stats['min']:.4f}")
                st.metric("Max Uncertainty", f"{unc_stats['max']:.4f}")

                st.write("**Error vs Uncertainty Correlation**")
                if actual_column and actual_column in merged.columns:
                    merged[f"{actual_column}_clean"] = pd.to_numeric(merged[actual_column], errors="coerce")
                    merged[f"{predicted_column}_clean"] = pd.to_numeric(merged[predicted_column], errors="coerce")
                    merged["error"] = (merged[f"{predicted_column}_clean"] - merged[f"{actual_column}_clean"]).abs()
                    corr = merged["uncertainty_built_up"].corr(merged["error"])
                    st.metric("Correlation", f"{corr:.4f}", help="Higher = uncertainty aligns with error")
        else:
            st.info("Uncertainty data not available for this model/split combination")
else:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Predictions Map")
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

# Predictions vs Truth Analysis Tab
st.markdown("---")
st.subheader("📊 Detailed Predictions vs Truth Analysis")

if actual_column and actual_column in filtered_predictions.columns:
    # Compute error metrics
    error_metrics = compute_error_metrics(
        filtered_predictions,
        predicted_column,
        actual_column,
    )

    # Display metrics in columns
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("R² Score", f"{error_metrics['R²']:.4f}", help="Higher is better (max 1.0)")
    with metric_cols[1]:
        st.metric("MAE", f"{error_metrics['MAE']:.4f}", help="Mean Absolute Error")
    with metric_cols[2]:
        st.metric("RMSE", f"{error_metrics['RMSE']:.4f}", help="Root Mean Squared Error")
    with metric_cols[3]:
        st.metric("MAPE", f"{error_metrics['MAPE']:.2f}%", help="Mean Absolute Percentage Error")

    st.markdown("---")

    # Visualization tabs
    viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5 = st.tabs([
        "📈 Predictions vs Truth",
        "📉 Residuals",
        "📊 Distributions",
        "🎯 Error Analysis",
        "📐 Q-Q Plot"
    ])

    with viz_tab1:
        st.write("**Scatter plot showing how well predictions match actual values.**")
        st.write("Points along the red dashed line indicate perfect predictions.")
        fig = plot_predictions_vs_truth(filtered_predictions, predicted_column, actual_column, selected_class)
        st.pyplot(fig)

    with viz_tab2:
        st.write("**Shows prediction errors across the range of predicted values.**")
        st.write("Ideally, points should be randomly scattered around zero (red line).")
        fig = plot_residuals(filtered_predictions, predicted_column, actual_column, selected_class)
        st.pyplot(fig)

    with viz_tab3:
        st.write("**Compares the distribution of predicted vs actual values.**")
        st.write("If distributions overlap well, the model captures the data range correctly.")
        fig = plot_distribution_comparison(filtered_predictions, predicted_column, actual_column, selected_class)
        st.pyplot(fig)

    with viz_tab4:
        st.write("**Distribution of prediction errors (residuals).**")
        st.write(f"Mean error: {error_metrics['Mean Error']:.4f} | Std: {error_metrics['Std Error']:.4f}")
        fig = plot_error_distribution(filtered_predictions, predicted_column, actual_column, selected_class)
        st.pyplot(fig)

    with viz_tab5:
        st.write("**Q-Q plot to assess if errors follow a normal distribution.**")
        st.write("Points close to the red line indicate normally distributed errors (good for regression).")
        fig = plot_quantile_quantile(filtered_predictions, predicted_column, actual_column, selected_class)
        st.pyplot(fig)

    st.markdown("---")

    # Detailed metrics table
    st.subheader("📋 Error Metrics Summary")
    metrics_df = pd.DataFrame(
        list(error_metrics.items()),
        columns=['Metric', 'Value']
    )
    metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.4f}")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

else:
    st.warning(
        "Predictions vs Truth analysis requires actual values to be present in the dataset. "
        "Change to composition mode if not already selected."
    )

# =============================================================================
# Composition Dashboard Sections
# =============================================================================
_render_composition_summary(filtered_predictions)
_render_uncertainty_section(filtered_predictions, merged)
_render_change_summary(filtered_predictions)
_render_error_analysis(filtered_predictions)
comparison_summary_frame = _render_model_comparison(metrics)
_render_dominant_summary(filtered_predictions)
_render_downloads(filtered_predictions, comparison_summary_frame)
