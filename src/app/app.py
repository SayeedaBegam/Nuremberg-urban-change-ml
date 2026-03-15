from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# Make the repository root importable when Streamlit runs the file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.explain_utils import helpful_explanation, misleading_explanation
from src.app.map_utils import build_map
from src.utils.config import INTERIM_DIR, MODELS_DIR, PROCESSED_DIR


st.set_page_config(page_title="Nuremberg Urban Change", layout="wide")
st.title("Mapping Urban Change in Nuremberg")
st.caption("Tabular machine learning on satellite-derived features and ESA WorldCover labels.")

dataset = pd.read_csv(PROCESSED_DIR / "change_dataset.csv")
predictions = pd.read_csv(PROCESSED_DIR / "app_predictions.csv") if (PROCESSED_DIR / "app_predictions.csv").exists() else None
grid = gpd.read_file(INTERIM_DIR / "grid.geojson").to_crs(4326)
base = grid.merge(dataset, on="cell_id", how="left")
merged = base.merge(predictions, on="cell_id", how="left") if predictions is not None else base

target = st.sidebar.selectbox(
    "Target layer",
    ["pred_delta_built_up", "pred_delta_vegetation", "pred_delta_water", "pred_delta_other"]
    if predictions is not None
    else ["delta_built_up", "delta_vegetation", "delta_water", "delta_other"],
)

st.sidebar.markdown("### Limits")
st.sidebar.write(
    "Use this dashboard for broad pattern exploration only. It is not suitable for parcel-level decisions."
)

metrics_path = MODELS_DIR / "metrics.json"
if metrics_path.exists():
    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)
    st.sidebar.markdown("### Evaluation")
    st.sidebar.json(metrics.get("forest", {}))

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Change map")
    tooltip_fields = ["cell_id", target, "change_binary"]
    if "uncertainty_built_up" in merged.columns:
        tooltip_fields.append("uncertainty_built_up")
    map_obj = build_map(merged, target, tooltip_fields)
    st_folium(map_obj, width=900, height=600)

with col2:
    st.subheader("Explanation")
    st.write(helpful_explanation())
    st.subheader("Potentially misleading explanation")
    st.write(misleading_explanation())
    st.subheader("Important limitations")
    st.write(
        "Predictions can be unreliable in mixed land-cover cells, boundary regions, cloudy imagery, and "
        "areas where label quality is weak."
    )
