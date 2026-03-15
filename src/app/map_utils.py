from __future__ import annotations

import folium
import geopandas as gpd


def build_map(gdf: gpd.GeoDataFrame, value_column: str, tooltip_columns: list[str]) -> folium.Map:
    center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
    fmap = folium.Map(location=center, zoom_start=11, tiles="cartodbpositron")

    folium.Choropleth(
        geo_data=gdf.to_json(),
        data=gdf,
        columns=["cell_id", value_column],
        key_on="feature.properties.cell_id",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=value_column,
    ).add_to(fmap)

    tooltip = folium.features.GeoJsonTooltip(fields=tooltip_columns)
    folium.GeoJson(gdf.to_json(), tooltip=tooltip).add_to(fmap)
    return fmap
