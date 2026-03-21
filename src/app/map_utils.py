from __future__ import annotations

import json

import folium
import geopandas as gpd
import pandas as pd
from branca.colormap import LinearColormap


COMPOSITION_COLORS = {
    "vegetation": ["#f7fcf5", "#c7e9c0", "#74c476", "#31a354", "#006d2c"],
    "built_up": ["#fff5f0", "#fcbba1", "#fb6a4a", "#de2d26", "#a50f15"],
    "water": ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
    "other": ["#f7f7f7", "#d9d9d9", "#bdbdbd", "#969696", "#636363"],
}
DIVERGING_COLORS = ["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"]


def _value_bounds(values: pd.Series, layer_mode: str) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return (0.0, 1.0) if layer_mode == "composition" else (-1.0, 1.0)

    if layer_mode == "change":
        limit = max(abs(float(numeric.min())), abs(float(numeric.max())), 1e-6)
        return -limit, limit

    vmin = max(0.0, float(numeric.min()))
    vmax = min(1.0, float(numeric.max()))
    if abs(vmax - vmin) < 1e-6:
        vmax = vmin + 1e-6
    return vmin, vmax


def _target_class(value_column: str) -> str:
    for class_name in COMPOSITION_COLORS:
        if class_name in value_column:
            return class_name
    return "other"


def _build_colormap(values: pd.Series, layer_mode: str, value_column: str) -> LinearColormap:
    vmin, vmax = _value_bounds(values, layer_mode)
    if layer_mode == "change":
        colors = DIVERGING_COLORS
    else:
        colors = COMPOSITION_COLORS[_target_class(value_column)]
    return LinearColormap(colors=colors, vmin=vmin, vmax=vmax)


def build_map(
    gdf: gpd.GeoDataFrame,
    value_column: str,
    tooltip_columns: list[str],
    layer_mode: str,
    legend_name: str,
) -> folium.Map:
    if gdf.empty:
        raise ValueError("Cannot build a map from an empty GeoDataFrame.")

    plot_gdf = gdf.copy()
    plot_gdf[value_column] = pd.to_numeric(plot_gdf[value_column], errors="coerce")
    minx, miny, maxx, maxy = plot_gdf.total_bounds
    center = [(miny + maxy) / 2.0, (minx + maxx) / 2.0]
    fmap = folium.Map(location=center, zoom_start=11, tiles="cartodbpositron")
    colormap = _build_colormap(plot_gdf[value_column], layer_mode, value_column)
    colormap.caption = legend_name

    def style_function(feature: dict) -> dict[str, object]:
        value = feature["properties"].get(value_column)
        if value is None or pd.isna(value):
            return {
                "fillColor": "#00000000",
                "color": "#9a9a9a",
                "weight": 0.12,
                "fillOpacity": 0.0,
            }
        return {
            "fillColor": colormap(value),
            "color": "#4d4d4d",
            "weight": 0.2,
            "fillOpacity": 0.82,
        }

    aliases = [column.replace("_", " ") for column in tooltip_columns]
    geojson = folium.GeoJson(
        data=json.loads(plot_gdf.to_json()),
        style_function=style_function,
        highlight_function=lambda _: {"weight": 1.0, "color": "#111111", "fillOpacity": 0.9},
        tooltip=folium.features.GeoJsonTooltip(
            fields=tooltip_columns,
            aliases=aliases,
            localize=True,
            sticky=False,
            labels=True,
        ),
    )
    geojson.add_to(fmap)
    colormap.add_to(fmap)
    return fmap
