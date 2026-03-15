from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd


def _safe_cell_area(geometry) -> float:
    area = geometry.area
    return area if area > 0 else np.nan


def build_osm_context_features(grid: gpd.GeoDataFrame, vector_path) -> pd.DataFrame:
    """Compute optional static road and building density features from local OSM exports."""
    osm = gpd.read_file(vector_path)
    if osm.empty:
        return pd.DataFrame({"cell_id": grid["cell_id"], "road_density": 0.0, "building_density": 0.0})

    osm = osm.to_crs(grid.crs)
    if "building" in osm.columns:
        buildings = osm[osm["building"].notna()].copy()
    else:
        buildings = osm.iloc[0:0].copy()

    if "highway" in osm.columns:
        roads = osm[osm["highway"].notna()].copy()
    else:
        roads = osm.iloc[0:0].copy()

    rows = []
    for record in grid.itertuples(index=False):
        cell_area = _safe_cell_area(record.geometry)
        road_density = 0.0
        building_density = 0.0

        if not roads.empty:
            road_parts = roads[roads.intersects(record.geometry)]
            if not road_parts.empty:
                road_density = road_parts.intersection(record.geometry).length.sum() / cell_area

        if not buildings.empty:
            building_parts = buildings[buildings.intersects(record.geometry)]
            if not building_parts.empty:
                building_density = building_parts.intersection(record.geometry).area.sum() / cell_area

        rows.append(
            {
                "cell_id": record.cell_id,
                "road_density": float(road_density),
                "building_density": float(building_density),
            }
        )

    return pd.DataFrame(rows)
