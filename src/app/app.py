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
    plot_predictions_vs_truth,
    plot_residuals,
    plot_distribution_comparison,
    plot_error_distribution,
    plot_quantile_quantile,
    compute_error_metrics,
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
LAND_COVER_CLASSES = ["built_up", "vegetation", "water", "other"]
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

    return _compute_delta_columns(predictions)


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
