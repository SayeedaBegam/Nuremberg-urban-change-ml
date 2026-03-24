from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import folium
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from branca.colormap import LinearColormap
from shapely.geometry import shape
from sklearn.metrics import r2_score
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
    compute_error_metrics,
)
from src.utils.config import CHANGE_TARGET_COLUMNS, INTERIM_DIR, MODELS_DIR, PROCESSED_DIR


APP_PREDICTIONS_PATH = PROCESSED_DIR / "app_predictions_combined.csv"
METRICS_PATH = MODELS_DIR / "metrics.json"
ENHANCED_METRICS_PATH = MODELS_DIR / "metrics_2020_2021_enhanced.json"
CHANGE_DATASET_PATH = PROCESSED_DIR / "change_dataset_2020_2021.csv"
ELASTIC_MODEL_PATH = MODELS_DIR / "elastic_net_2020_2021_enhanced.joblib"
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


def _apply_dark_visual_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp,
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(1200px 620px at 15% -10%, #172445 0%, #0a1020 48%, #070d1a 100%);
            color: #e6edf8;
        }
        [data-testid="stHeader"] {
            background: rgba(7, 13, 26, 0.7);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f1a31 0%, #0a1328 100%);
            border-right: 1px solid #1f2e4a;
        }
        div[data-testid="stMetric"] {
            background: rgba(21, 34, 56, 0.66);
            border: 1px solid #1f3558;
            border-radius: 8px;
            padding: 0.7rem 0.8rem;
        }
        div[data-testid="stDataFrame"] div[role="table"] {
            border: 1px solid #1f3558;
            border-radius: 8px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(28, 43, 69, 0.7);
            border: 1px solid #284066;
            border-radius: 8px;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(52, 86, 132, 0.82);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _style_dark_chart(chart: alt.Chart | alt.LayerChart) -> alt.Chart | alt.LayerChart:
    return chart.configure_view(stroke=None).configure_axis(
        gridColor="#21314d",
        domainColor="#36527a",
        labelColor="#dbe8ff",
        titleColor="#dbe8ff",
    ).configure_legend(
        labelColor="#dbe8ff",
        titleColor="#dbe8ff",
    ).configure_title(color="#dbe8ff")


_apply_dark_visual_theme()


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

    # Add elastic_net rows from saved model artifact when available so model selection can switch between both.
    reference_splits = tuple(sorted(predictions["split"].dropna().astype(str).unique().tolist()))
    elastic_rows = _elastic_predictions_from_artifacts(reference_splits)
    if not elastic_rows.empty:
        elastic_rows["cell_id"] = elastic_rows["cell_id"].astype(str)
        for column in predictions.columns:
            if column not in elastic_rows.columns:
                elastic_rows[column] = np.nan
        for column in elastic_rows.columns:
            if column not in predictions.columns:
                predictions[column] = np.nan
        predictions = pd.concat([predictions, elastic_rows[predictions.columns]], ignore_index=True)

    return prepare_dashboard_frame(predictions)


def _load_metrics() -> dict:
    if METRICS_PATH.exists():
        with METRICS_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def _load_enhanced_metrics() -> dict:
    if not ENHANCED_METRICS_PATH.exists():
        return {}
    with ENHANCED_METRICS_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


@st.cache_data(show_spinner=False)
def _load_change_dataset() -> pd.DataFrame:
    if not CHANGE_DATASET_PATH.exists():
        return pd.DataFrame()
    frame = pd.read_csv(CHANGE_DATASET_PATH)
    if "cell_id" in frame.columns:
        frame["cell_id"] = frame["cell_id"].astype(str)
    return frame


@st.cache_data(show_spinner=False)
def _elastic_predictions_from_artifacts(reference_splits: tuple[str, ...]) -> pd.DataFrame:
    if not ELASTIC_MODEL_PATH.exists():
        return pd.DataFrame()

    change_data = _load_change_dataset()
    if change_data.empty:
        return pd.DataFrame()

    try:
        from src.models.train_2020_2021_enhanced import _create_enhanced_features, _feature_columns
        from src.models.uncertainty import elastic_net_uncertainty
    except Exception:
        return pd.DataFrame()

    feature_ready = _create_enhanced_features(change_data.copy())
    feature_columns = [column for column in _feature_columns(feature_ready) if column in feature_ready.columns]
    if not feature_columns:
        return pd.DataFrame()

    try:
        model = joblib.load(ELASTIC_MODEL_PATH)
        predictions = model.predict(feature_ready[feature_columns].fillna(0))
    except Exception:
        return pd.DataFrame()

    pred_frame = pd.DataFrame(predictions, columns=CHANGE_TARGET_COLUMNS)
    export = change_data[["cell_id"]].copy()

    for target in CHANGE_TARGET_COLUMNS:
        class_name = target.replace("delta_", "")
        if target not in change_data.columns:
            return pd.DataFrame()
        export[f"actual_delta_{class_name}"] = pd.to_numeric(change_data[target], errors="coerce")
        export[f"pred_delta_{class_name}"] = pd.to_numeric(pred_frame[target], errors="coerce")

    if "centroid_x_t1" in change_data.columns:
        export["centroid_x_t1"] = change_data["centroid_x_t1"]
    if "centroid_y_t1" in change_data.columns:
        export["centroid_y_t1"] = change_data["centroid_y_t1"]
    
    # Calculate uncertainty based on residuals
    actual_built_up = pd.to_numeric(change_data.get("delta_built_up", pd.Series()), errors="coerce")
    pred_built_up = pd.to_numeric(pred_frame.get("delta_built_up", pd.Series()), errors="coerce")
    
    # Only calculate uncertainty where we have valid predictions
    valid_mask = actual_built_up.notna() & pred_built_up.notna()
    uncertainty = np.full(len(export), np.nan)
    if valid_mask.any():
        uncertainty[valid_mask] = elastic_net_uncertainty(
            actual_built_up[valid_mask].values,
            pred_built_up[valid_mask].values
        )
    
    export["uncertainty_built_up"] = uncertainty
    export["model"] = "elastic_net"
    export["split"] = reference_splits[0] if reference_splits else "full_2020_2021_enhanced"
    return export


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


def _build_analysis_frame(frame: pd.DataFrame, predicted_column: str, actual_column: str | None) -> pd.DataFrame:
    if actual_column is None or actual_column not in frame.columns or predicted_column not in frame.columns:
        return pd.DataFrame()

    columns = ["cell_id", predicted_column, actual_column]
    if "uncertainty_built_up" in frame.columns:
        columns.append("uncertainty_built_up")

    analysis = frame[columns].copy()
    analysis = analysis.rename(columns={predicted_column: "predicted", actual_column: "actual"})
    analysis["predicted"] = pd.to_numeric(analysis["predicted"], errors="coerce")
    analysis["actual"] = pd.to_numeric(analysis["actual"], errors="coerce")
    analysis = analysis.dropna(subset=["predicted", "actual"])
    if analysis.empty:
        return pd.DataFrame()

    analysis["error"] = analysis["predicted"] - analysis["actual"]
    analysis["abs_error"] = analysis["error"].abs()
    if "uncertainty_built_up" in analysis.columns:
        analysis["uncertainty"] = pd.to_numeric(analysis["uncertainty_built_up"], errors="coerce")
    else:
        analysis["uncertainty"] = np.nan
    return analysis


def _class_delta_columns(frame: pd.DataFrame, class_name: str) -> tuple[str | None, str | None]:
    predicted = None
    for candidate in [f"pred_delta_{class_name}", f"pred_delta_{class_name}_random_forest", f"pred_delta_{class_name}_rf"]:
        if candidate in frame.columns:
            predicted = candidate
            break
    actual = f"actual_delta_{class_name}" if f"actual_delta_{class_name}" in frame.columns else None
    return predicted, actual


def _render_per_cell_change_analysis(frame: pd.DataFrame, selected_class: str, predicted_column: str, actual_column: str | None) -> None:
    st.divider()
    st.header("Per-Cell Change Analysis")
    analysis = _build_analysis_frame(frame, predicted_column, actual_column)
    if analysis.empty:
        st.info("Per-cell analysis is unavailable because predicted and actual values are missing for this selection.")
        return

    correlation = float(analysis["predicted"].corr(analysis["actual"]))
    mean_error = float(analysis["error"].mean())
    mae = float(analysis["abs_error"].mean())

    metric_cols = st.columns(4)
    metric_cols[0].metric("Total cells", f"{len(analysis):,}")
    metric_cols[1].metric("Correlation", f"{correlation:.4f}")
    metric_cols[2].metric("Mean error", f"{mean_error:.4f}")
    metric_cols[3].metric("MAE", f"{mae:.4f}")

    lower = float(min(analysis["actual"].min(), analysis["predicted"].min()))
    upper = float(max(analysis["actual"].max(), analysis["predicted"].max()))
    guide = pd.DataFrame({"actual": [lower, upper], "predicted": [lower, upper]})
    scatter = alt.Chart(analysis).mark_circle(size=35, opacity=0.65).encode(
        x=alt.X("actual:Q", title="Actual change"),
        y=alt.Y("predicted:Q", title="Predicted change"),
        color=alt.Color("abs_error:Q", scale=alt.Scale(scheme="reds"), title="Abs error"),
        tooltip=["cell_id", alt.Tooltip("actual:Q", format=".4f"), alt.Tooltip("predicted:Q", format=".4f"), alt.Tooltip("abs_error:Q", format=".4f")],
    )
    diagonal = alt.Chart(guide).mark_line(color="#ff4d4f", strokeDash=[8, 4]).encode(x="actual:Q", y="predicted:Q")
    error_hist = alt.Chart(analysis).mark_bar(color="#7bb6e5").encode(
        x=alt.X("error:Q", bin=alt.Bin(maxbins=42), title="Prediction error (predicted - actual)"),
        y=alt.Y("count():Q", title="Cells"),
        tooltip=[alt.Tooltip("count():Q", title="Cells")],
    )

    chart_cols = st.columns(2)
    chart_cols[0].altair_chart(
        _style_dark_chart((scatter + diagonal).properties(height=320, title=f"Predicted vs Actual: {selected_class}")),
        use_container_width=True,
    )
    chart_cols[1].altair_chart(
        _style_dark_chart(error_hist.properties(height=320, title="Error Distribution (per cell)")),
        use_container_width=True,
    )

    density_source = analysis[["predicted", "actual"]].rename(columns={"predicted": "Predicted", "actual": "Actual"}).melt(
        var_name="series", value_name="value"
    )
    density_chart = (
        alt.Chart(density_source)
        .transform_density("value", groupby=["series"], as_=["value", "density"])
        .mark_area(opacity=0.42)
        .encode(
            x=alt.X("value:Q", title="Change value"),
            y=alt.Y("density:Q", title="Density"),
            color=alt.Color("series:N", scale=alt.Scale(domain=["Actual", "Predicted"], range=["#85c4ff", "#f39a63"])),
            tooltip=["series", alt.Tooltip("value:Q", format=".4f"), alt.Tooltip("density:Q", format=".4f")],
        )
    )

    lower_cols = st.columns(2)
    lower_cols[0].altair_chart(
        _style_dark_chart(density_chart.properties(height=220, title="Distribution Comparison")),
        use_container_width=True,
    )
    if analysis["uncertainty"].notna().any():
        uncertainty_frame = analysis.dropna(subset=["uncertainty"])
        uncertainty_chart = alt.Chart(uncertainty_frame).mark_circle(size=28, opacity=0.58, color="#8ad0ff").encode(
            x=alt.X("uncertainty:Q", title="Model uncertainty"),
            y=alt.Y("abs_error:Q", title="Absolute error"),
            tooltip=["cell_id", alt.Tooltip("uncertainty:Q", format=".4f"), alt.Tooltip("abs_error:Q", format=".4f")],
        )
        lower_cols[1].altair_chart(
            _style_dark_chart(uncertainty_chart.properties(height=220, title="Error vs Uncertainty (per cell)")),
            use_container_width=True,
        )
        unc_corr = float(uncertainty_frame["uncertainty"].corr(uncertainty_frame["abs_error"]))
        st.caption(f"Uncertainty-Error Correlation: {unc_corr:.4f} (higher indicates uncertainty aligns with absolute error)")
    else:
        lower_cols[1].info("Uncertainty column is unavailable for this selection.")

    table_columns = ["cell_id", "predicted", "actual", "error", "abs_error"]
    if analysis["uncertainty"].notna().any():
        table_columns.append("uncertainty")
    highest_errors = analysis.loc[:, table_columns].sort_values("abs_error", ascending=False).head(12).reset_index(drop=True)
    st.subheader("Cells with Highest Prediction Errors")
    st.dataframe(highest_errors, use_container_width=True)

    st.subheader("Land-Cover Change Summary (All Classes)")
    summary_rows = []
    for class_name in LAND_COVER_CLASSES:
        pred_col, act_col = _class_delta_columns(frame, class_name)
        if pred_col is not None:
            summary_rows.append({"class": class_name, "source": "Predicted", "value": float(pd.to_numeric(frame[pred_col], errors="coerce").mean())})
        if act_col is not None:
            summary_rows.append({"class": class_name, "source": "Actual", "value": float(pd.to_numeric(frame[act_col], errors="coerce").mean())})

    if not summary_rows:
        st.info("Class-level change summary cannot be shown because delta columns are missing.")
        return

    summary_frame = pd.DataFrame(summary_rows)
    summary_bar = alt.Chart(summary_frame).mark_bar().encode(
        x=alt.X("class:N", title=None),
        xOffset=alt.XOffset("source:N"),
        y=alt.Y("value:Q", title="Average change"),
        color=alt.Color("source:N", scale=alt.Scale(domain=["Actual", "Predicted"], range=["#7bb6e5", "#1f78d1"])),
        tooltip=["class", "source", alt.Tooltip("value:Q", format=".4f")],
    )

    summary_cols = st.columns([1.45, 1.05])
    summary_cols[0].altair_chart(_style_dark_chart(summary_bar.properties(height=260)), use_container_width=True)

    pie_cols = summary_cols[1].columns(2)
    predicted_positive = positive_change_share(frame, "pred_delta_{class_name}")
    actual_positive = positive_change_share(frame, "actual_delta_{class_name}")
    with pie_cols[0]:
        if not predicted_positive.empty and predicted_positive["value"].sum() > 0:
            st.altair_chart(_style_dark_chart(pie_chart(predicted_positive, "Predicted positive change")), use_container_width=True)
        else:
            st.info("No positive predicted change.")
    with pie_cols[1]:
        if not actual_positive.empty and actual_positive["value"].sum() > 0:
            st.altair_chart(_style_dark_chart(pie_chart(actual_positive, "Actual positive change")), use_container_width=True)
        else:
            st.info("No positive actual change.")


def _render_change_error_matrix(analysis: pd.DataFrame) -> None:
    st.divider()
    st.header("Change Error Analysis")
    if analysis.empty:
        st.info("Change error analysis is unavailable for the current selection.")
        return

    max_threshold = float(
        max(
            analysis["actual"].abs().quantile(0.995),
            analysis["predicted"].abs().quantile(0.995),
            1e-3,
        )
    )
    threshold_default = float(np.clip(analysis["actual"].abs().quantile(0.7), 0.0005, max_threshold))
    step_size = max(round(max_threshold / 250.0, 4), 0.0005)
    threshold = st.slider(
        "Decision threshold (absolute change)",
        min_value=0.0,
        max_value=float(round(max_threshold, 4)),
        value=float(round(threshold_default, 4)),
        step=float(step_size),
        key="change_threshold",
    )

    true_change = (analysis["actual"].abs() >= threshold).astype(int)
    predicted_change = (analysis["predicted"].abs() >= threshold).astype(int)

    tp = int(((true_change == 1) & (predicted_change == 1)).sum())
    tn = int(((true_change == 0) & (predicted_change == 0)).sum())
    fp = int(((true_change == 0) & (predicted_change == 1)).sum())
    fn = int(((true_change == 1) & (predicted_change == 0)).sum())
    total = max(len(analysis), 1)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total

    heatmap_rows = [
        {"actual": "No change", "predicted": "No change", "count": tn, "ratio": tn / total},
        {"actual": "No change", "predicted": "Change", "count": fp, "ratio": fp / total},
        {"actual": "Change", "predicted": "No change", "count": fn, "ratio": fn / total},
        {"actual": "Change", "predicted": "Change", "count": tp, "ratio": tp / total},
    ]
    heatmap = pd.DataFrame(heatmap_rows)
    matrix = alt.Chart(heatmap).mark_rect().encode(
        x=alt.X("predicted:N", title="Predicted"),
        y=alt.Y("actual:N", title="Actual"),
        color=alt.Color("ratio:Q", title="Cell share", scale=alt.Scale(scheme="blues")),
        tooltip=["actual", "predicted", "count", alt.Tooltip("ratio:Q", format=".3f")],
    )
    labels = alt.Chart(heatmap).mark_text(fontSize=14, color="#f3f8ff").encode(
        x="predicted:N",
        y="actual:N",
        text=alt.Text("count:Q", format=",.0f"),
    )

    left, right = st.columns([2.6, 1.4])
    left.altair_chart(_style_dark_chart((matrix + labels).properties(height=200)), use_container_width=True)
    with right:
        st.metric("Precision", f"{precision:.3f}")
        st.metric("Recall", f"{recall:.3f}")
        st.metric("F1 score", f"{f1:.3f}")
        st.metric("Accuracy", f"{accuracy:.3f}")

    counts = st.columns(4)
    counts[0].metric("TP", f"{tp:,}")
    counts[1].metric("Predicted change", f"{int(predicted_change.sum()):,}")
    counts[2].metric("TN", f"{tn:,}")
    counts[3].metric("FP", f"{fp:,}")


def _feature_impact_frame(change_data: pd.DataFrame, target_column: str, feature_candidates: list[str]) -> pd.DataFrame:
    if target_column not in change_data.columns:
        return pd.DataFrame()

    target = pd.to_numeric(change_data[target_column], errors="coerce")
    rows = []
    for feature in feature_candidates:
        if feature not in change_data.columns:
            continue
        values = pd.to_numeric(change_data[feature], errors="coerce")
        corr = values.corr(target)
        if pd.notna(corr):
            rows.append({"feature": feature, "coefficient": float(corr), "abs_value": abs(float(corr))})
    return pd.DataFrame(rows)


def _render_elastic_net_interpretability(selected_class: str, enhanced_metrics: dict, change_data: pd.DataFrame) -> None:
    st.divider()
    st.header("Elastic Net Interpretability")
    st.caption("Visual feature effects are approximated with feature-target correlations to mirror the decomposition layout.")

    if change_data.empty:
        st.info("Change dataset is unavailable, so interpretability visuals cannot be rendered.")
        return

    target_column = f"delta_{selected_class}"
    features_used = enhanced_metrics.get("features_used", [])
    if not features_used:
        features_used = [column for column in change_data.columns if column.endswith("_t1") or column.endswith("_t2")]
    impact_frame = _feature_impact_frame(change_data, target_column, features_used)
    if impact_frame.empty:
        st.info("No feature correlation signals are available for this class.")
        return

    top_negative = impact_frame.nsmallest(6, "coefficient")
    top_positive = impact_frame.nlargest(6, "coefficient")
    top_effects = pd.concat([top_negative, top_positive], ignore_index=True).drop_duplicates("feature")
    top_effects["direction"] = np.where(top_effects["coefficient"] >= 0, "Positive", "Negative")
    top_effects = top_effects.sort_values("coefficient")

    feature_chart = alt.Chart(top_effects).mark_bar().encode(
        x=alt.X("coefficient:Q", title="Correlation proxy"),
        y=alt.Y("feature:N", sort=alt.SortField(field="coefficient", order="ascending"), title=None),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["Negative", "Positive"], range=["#ff4d4f", "#36b36f"])),
        tooltip=["feature", alt.Tooltip("coefficient:Q", format=".4f")],
    )

    class_prefix = f"delta_{selected_class}"
    elastic = enhanced_metrics.get("elastic_net", {})
    forest = enhanced_metrics.get("random_forest", {})
    summary_rows = []
    for metric in ["r2", "mae", "rmse"]:
        elastic_value = elastic.get(f"{class_prefix}_{metric}")
        forest_value = forest.get(f"{class_prefix}_{metric}")
        if elastic_value is not None or forest_value is not None:
            summary_rows.append(
                {
                    "metric": metric.upper(),
                    "elastic_net": elastic_value,
                    "random_forest": forest_value,
                }
            )
    summary_table = pd.DataFrame(summary_rows)
    ranking_table = impact_frame.sort_values("abs_value", ascending=False).head(10)[["feature", "coefficient"]]
    ranking_table = ranking_table.rename(columns={"coefficient": "impact_proxy"})

    left, right = st.columns([1.2, 1.4])
    left.altair_chart(_style_dark_chart(feature_chart.properties(height=340)), use_container_width=True)
    with right:
        if not summary_table.empty:
            st.write("Model summary")
            st.dataframe(summary_table, use_container_width=True, hide_index=True)
        st.write("Top feature impact (proxy)")
        st.dataframe(ranking_table, use_container_width=True, hide_index=True)


def _build_square_noise_map(map_frame: gpd.GeoDataFrame, cell_radius: int) -> folium.Map:
    valid = map_frame.dropna(subset=["noise_shift"]).copy()
    if valid.empty:
        raise ValueError("Noise shift map is empty.")

    centers = valid.geometry.centroid
    center_lat = float(centers.y.mean())
    center_lon = float(centers.x.mean())
    noise_map = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="cartodbpositron")

    vmax = float(max(valid["noise_shift"].max(), 1e-6))
    colormap = LinearColormap(
        colors=["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
        vmin=0.0,
        vmax=vmax,
    )
    colormap.caption = "Noise impact | abs shift"
    colormap.add_to(noise_map)

    for row, center in zip(valid.itertuples(index=False), centers):
        folium.RegularPolygonMarker(
            location=[float(center.y), float(center.x)],
            number_of_sides=4,
            radius=cell_radius,
            rotation=45,
            color="#10223b",
            weight=0.7,
            fill=True,
            fill_color=colormap(float(row.noise_shift)),
            fill_opacity=0.92,
            tooltip=f"cell_id: {row.cell_id} | noise_shift: {float(row.noise_shift):.4f}",
        ).add_to(noise_map)

    return noise_map


def _render_noise_stress_test(
    analysis: pd.DataFrame,
    merged: gpd.GeoDataFrame,
    change_data: pd.DataFrame,
    enhanced_metrics: dict,
) -> None:
    st.divider()
    st.header("Stress Test: Feature Noise Robustness")
    if analysis.empty:
        st.info("Stress test is unavailable because the selected view does not include predicted and actual values.")
        return

    noise_level = st.slider("Noise level", min_value=0.0, max_value=0.35, value=0.08, step=0.01, key="stress_noise")
    scenario = st.selectbox("Scenario", ["Single feature", "All features"], key="stress_scenario")
    score_metric = st.selectbox("Score metric", ["R2", "MAE"], key="stress_metric")
    marker_radius = st.slider("Square cell radius", min_value=4, max_value=13, value=8, step=1, key="stress_marker_radius")

    feature_options: list[str] = []
    if not change_data.empty:
        metric_features = enhanced_metrics.get("features_used", [])
        if isinstance(metric_features, list):
            feature_options = [feature for feature in metric_features if feature in change_data.columns]
        if not feature_options:
            feature_options = [column for column in change_data.columns if column.endswith("_t1") or column.endswith("_t2")]

    selected_feature = None
    single_feature_table = pd.DataFrame()
    if scenario == "Single feature":
        if feature_options:
            selected_feature = st.selectbox("Feature", feature_options, key="stress_single_feature")
        else:
            st.info("No feature columns available for single-feature stress testing. Falling back to all-features noise.")
            scenario = "All features"

    feature_weight = np.ones(len(analysis), dtype=float)
    if scenario == "Single feature" and selected_feature and "cell_id" in change_data.columns:
        lookup = change_data[["cell_id", selected_feature]].copy()
        lookup["cell_id"] = lookup["cell_id"].astype(str)
        joined = analysis[["cell_id"]].merge(lookup, on="cell_id", how="left")
        feature_values = pd.to_numeric(joined[selected_feature], errors="coerce")
        if feature_values.notna().any():
            centered = feature_values - feature_values.mean()
            scale = feature_values.std(ddof=0) + 1e-9
            # Higher absolute standardized values get stronger perturbation in single-feature mode.
            standardized = np.abs(centered / scale)
            feature_weight = np.clip(standardized, 0.2, 2.8).fillna(1.0).to_numpy(dtype=float)
            single_feature_table = pd.DataFrame(
                {
                    "cell_id": joined["cell_id"],
                    "feature": selected_feature,
                    "feature_value": feature_values,
                    "feature_weight": feature_weight,
                }
            )

    rng = np.random.default_rng(42)
    predicted = analysis["predicted"].to_numpy(dtype=float)
    actual = analysis["actual"].to_numpy(dtype=float)
    std_base = float(np.std(predicted)) if len(predicted) else 0.0
    noise_std = max(std_base * noise_level, 1e-9)
    base_noise = rng.normal(0.0, noise_std, size=len(predicted))
    if scenario == "Single feature":
        noisy_predicted = predicted + base_noise * feature_weight
    else:
        noisy_predicted = predicted + base_noise

    finite_mask = np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(noisy_predicted)
    if not finite_mask.all():
        st.info(f"Filtered {int((~finite_mask).sum())} non-finite rows from stress-test scoring.")

    metric_actual = actual[finite_mask]
    metric_pred = predicted[finite_mask]
    metric_noisy = noisy_predicted[finite_mask]
    if len(metric_actual) < 2:
        st.warning("Not enough valid rows to compute stress-test metrics after filtering non-finite values.")
        return

    baseline_r2 = float(r2_score(metric_actual, metric_pred))
    noisy_r2 = float(r2_score(metric_actual, metric_noisy))
    baseline_mae = float(np.mean(np.abs(metric_pred - metric_actual)))
    noisy_mae = float(np.mean(np.abs(metric_noisy - metric_actual)))

    if score_metric == "R2":
        baseline_score = baseline_r2
        noisy_score = noisy_r2
        impact = noisy_score - baseline_score
    else:
        baseline_score = baseline_mae
        noisy_score = noisy_mae
        impact = baseline_score - noisy_score

    metrics_cols = st.columns(3)
    metrics_cols[0].metric("Base score", f"{baseline_score:.3f}")
    metrics_cols[1].metric("Noisy score", f"{noisy_score:.3f}")
    metrics_cols[2].metric("Score impact", f"{impact:.3f}")

    shift_frame = analysis[["cell_id"]].copy()
    shift_frame["noise_shift"] = np.abs(noisy_predicted - predicted)
    shift_hist = alt.Chart(shift_frame).mark_bar(color="#7fb8e6").encode(
        x=alt.X("noise_shift:Q", bin=alt.Bin(maxbins=28), title="Absolute prediction shift"),
        y=alt.Y("count():Q", title="Cells"),
        tooltip=[alt.Tooltip("count():Q", title="Cells")],
    )

    lower = st.columns([1.4, 1.2])
    lower[0].altair_chart(_style_dark_chart(shift_hist.properties(height=260)), use_container_width=True)

    with lower[1]:
        map_frame = merged[["cell_id", "geometry"]].copy().merge(shift_frame, on="cell_id", how="left")
        if map_frame["noise_shift"].notna().any():
            stress_map = _build_square_noise_map(map_frame, cell_radius=marker_radius)
            st_folium(stress_map, width=520, height=330)
        else:
            st.info("Noise impact map is unavailable for this selection.")

    if scenario == "Single feature" and selected_feature and not single_feature_table.empty:
        table = single_feature_table.merge(shift_frame, on="cell_id", how="left")
        table = table.sort_values("feature_weight", ascending=False).head(20).reset_index(drop=True)
        st.subheader("Single Feature Stress Table")
        st.dataframe(table, use_container_width=True, hide_index=True)


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

if not available_models:
    st.error("No model values were found in the prediction export.")
    st.stop()
if not available_splits:
    st.error("No split values were found in the prediction export.")
    st.stop()

st.sidebar.markdown("### Controls")
selected_layer_mode = st.sidebar.radio(
    "Change / Composition",
    options=LAYER_MODES,
    index=1 if "change" in LAYER_MODES else 0,
    format_func=lambda value: value.title(),
)
selected_model = st.sidebar.selectbox("Model selection", available_models)
selected_split = st.sidebar.selectbox("Data selection", available_splits)
selected_class = st.sidebar.selectbox("Class", LAND_COVER_CLASSES, index=0)

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

effective_layer_mode = selected_layer_mode
if selected_layer_mode == "composition":
    predicted_column = f"pred_{selected_class}_prop_t2"
    actual_column = f"actual_{selected_class}_prop_t2"
    if predicted_column not in filtered_predictions.columns:
        st.warning(
            "Composition columns are unavailable in this dataset export. Showing change mode for the selected class."
        )
        effective_layer_mode = "change"
        predicted_column, actual_column = _get_available_columns(filtered_predictions, selected_class)
        if predicted_column is None:
            st.error(f"No predicted delta columns found for {selected_class}.")
            st.stop()
elif selected_layer_mode == "change":
    predicted_column, actual_column = _get_available_columns(filtered_predictions, selected_class)
    if predicted_column is None:
        st.error(f"No predicted delta columns found for {selected_class}.")
        st.stop()
else:
    st.error(f"Unsupported layer mode: {selected_layer_mode}")
    st.stop()

if predicted_column not in filtered_predictions.columns:
    st.error(f"Missing value column `{predicted_column}` in `{APP_PREDICTIONS_PATH}`.")
    st.stop()

merged = grid.merge(filtered_predictions, on="cell_id", how="left")
predicted_tooltip_columns = _build_tooltip_columns(merged, predicted_column, actual_column)
legend_name = f"{selected_model} | {selected_split} | {effective_layer_mode} | {selected_class}"

# Show map and explanation
# For change mode with actual data, show side-by-side comparison; otherwise show single map
show_comparison = (effective_layer_mode == "change" and
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
                layer_mode=effective_layer_mode,
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
                layer_mode=effective_layer_mode,
                legend_name=f"Actual {selected_class} change",
            )
            actual_map.fit_bounds(map_bounds)
            st_folium(actual_map, width=700, height=650)

        with col3:
            st.subheader("Explanation")
            st.write(helpful_explanation(effective_layer_mode, selected_class))
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
            layer_mode=effective_layer_mode,
            legend_name=legend_name,
        )
        st_folium(map_obj, width=900, height=600)

    with col2:
        st.subheader("Explanation")
        st.write(helpful_explanation(effective_layer_mode, selected_class))
        st.subheader("Potentially Misleading Explanation")
        st.write(misleading_explanation())
        st.subheader("Important Limitations")
        st.write(limitations_text())

enhanced_metrics_payload = _load_enhanced_metrics()
change_dataset = _load_change_dataset()
analysis_frame = _build_analysis_frame(filtered_predictions, predicted_column, actual_column)

_render_per_cell_change_analysis(filtered_predictions, selected_class, predicted_column, actual_column)
_render_change_error_matrix(analysis_frame)
_render_elastic_net_interpretability(selected_class, enhanced_metrics_payload, change_dataset)
_render_noise_stress_test(analysis_frame, merged, change_dataset, enhanced_metrics_payload)

# Compact metrics section to avoid image-cache warnings from large static matplotlib renders.
st.markdown("---")
st.subheader("Detailed Predictions vs Truth Metrics")
if actual_column and actual_column in filtered_predictions.columns:
    error_metrics = compute_error_metrics(filtered_predictions, predicted_column, actual_column)
    metrics_df = pd.DataFrame(list(error_metrics.items()), columns=["Metric", "Value"])
    metrics_df["Value"] = metrics_df["Value"].map(lambda value: float(value))
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
else:
    st.info("Detailed metrics are unavailable for this selection because actual values are missing.")

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
