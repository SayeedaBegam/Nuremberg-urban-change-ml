from pathlib import Path
import geopandas as gpd
import pandas as pd

# -----------------------------
# PATHS
# -----------------------------
GRID_PATH = Path("data/processed_esa_ee_osm/nuremberg_grid_250m_wgs84.geojson")
ROADS_PATH = Path("data/osm/2021/gis_osm_roads_free_1.shp")
BUILDINGS_PATH = Path("data/osm/2021/gis_osm_buildings_a_free_1.shp")
OUT_PATH = Path("data/processed_esa_ee_osm/nuremberg_2021_osm_features_250m.csv")


# -----------------------------
# LOADERS
# -----------------------------
def load_grid_gdf(path, to_crs=None):
    gdf = gpd.read_file(path)
    gdf = gdf[(gdf.geometry.notna()) & (~gdf.geometry.is_empty)].copy()

    if gdf.crs is None:
        gdf = gdf.set_crs(4326)

    if to_crs is not None:
        gdf = gdf.to_crs(to_crs)

    return gdf


def load_roads_gdf(path, to_crs=None):
    gdf = gpd.read_file(path)
    gdf = gdf[(gdf.geometry.notna()) & (~gdf.geometry.is_empty)].copy()

    if gdf.crs is None:
        gdf = gdf.set_crs(4326)

    if to_crs is not None:
        gdf = gdf.to_crs(to_crs)

    # NO buffer(0) here for roads
    return gdf


def load_buildings_gdf(path, to_crs=None):
    gdf = gpd.read_file(path)
    gdf = gdf[(gdf.geometry.notna()) & (~gdf.geometry.is_empty)].copy()

    if gdf.crs is None:
        gdf = gdf.set_crs(4326)

    if to_crs is not None:
        gdf = gdf.to_crs(to_crs)

    # Repair only polygon geometries
    def safe_fix(geom):
        try:
            return geom.buffer(0)
        except Exception:
            return None

    gdf["geometry"] = gdf.geometry.apply(safe_fix)
    gdf = gdf[(gdf.geometry.notna()) & (~gdf.geometry.is_empty)].copy()

    return gdf


# -----------------------------
# 1. LOAD DATA
# -----------------------------
grid = load_grid_gdf(GRID_PATH, to_crs=32632)
roads = load_roads_gdf(ROADS_PATH, to_crs=32632)
buildings = load_buildings_gdf(BUILDINGS_PATH, to_crs=32632)

# Keep only expected geometry types
roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])].copy()
roads = roads[["geometry"]].copy()

buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
buildings = buildings[["geometry"]].copy()

# -----------------------------
# 2. DEBUG CHECKS
# -----------------------------
print("Grid rows:", len(grid))
print("Road rows:", len(roads))
print("Building rows:", len(buildings))

print("Grid CRS:", grid.crs)
print("Roads CRS:", roads.crs)
print("Buildings CRS:", buildings.crs)

print("Grid bounds:", grid.total_bounds)
print("Roads bounds:", roads.total_bounds)
print("Buildings bounds:", buildings.total_bounds)

print(grid.head())

# Cell area
grid["cell_area_m2"] = grid.geometry.area

# -----------------------------
# 3. ROAD LENGTH PER CELL
# -----------------------------
road_intersections = gpd.overlay(
    grid[["cell_id", "geometry"]],
    roads,
    how="intersection",
    keep_geom_type=False
)

road_intersections = road_intersections[
    (road_intersections.geometry.notna()) & (~road_intersections.geometry.is_empty)
].copy()

print("Road intersections rows:", len(road_intersections))

road_intersections["road_length_m"] = road_intersections.geometry.length

road_by_cell = (
    road_intersections.groupby("cell_id", as_index=False)["road_length_m"]
    .sum()
)

# -----------------------------
# 4. BUILDING AREA PER CELL
# -----------------------------
building_intersections = gpd.overlay(
    grid[["cell_id", "geometry"]],
    buildings,
    how="intersection",
    keep_geom_type=False
)

building_intersections = building_intersections[
    (building_intersections.geometry.notna()) & (~building_intersections.geometry.is_empty)
].copy()

building_intersections = building_intersections[
    building_intersections.geometry.type.isin(["Polygon", "MultiPolygon"])
].copy()

print("Building intersections rows:", len(building_intersections))

building_intersections["building_area_m2"] = building_intersections.geometry.area

building_by_cell = (
    building_intersections.groupby("cell_id", as_index=False)["building_area_m2"]
    .sum()
)

# -----------------------------
# 5. MERGE + DERIVE FEATURES
# -----------------------------
osm_features = grid[["cell_id", "cell_area_m2"]].copy()

osm_features = osm_features.merge(road_by_cell, on="cell_id", how="left")
osm_features = osm_features.merge(building_by_cell, on="cell_id", how="left")

osm_features["road_length_m"] = osm_features["road_length_m"].fillna(0.0)
osm_features["building_area_m2"] = osm_features["building_area_m2"].fillna(0.0)

osm_features["road_density"] = osm_features["road_length_m"] / osm_features["cell_area_m2"]
osm_features["building_area_ratio"] = osm_features["building_area_m2"] / osm_features["cell_area_m2"]

osm_features = osm_features[["cell_id", "road_density", "building_area_ratio"]]

# -----------------------------
# 6. CHECK + SAVE
# -----------------------------
print("\nOSM features preview:")
print(osm_features.head())

print("\nShape:")
print(osm_features.shape)

print("\nNull counts:")
print(osm_features.isnull().sum())

print("\nSummary stats:")
print(osm_features.describe())

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
osm_features.to_csv(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}")