from pathlib import Path
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# PATHS
# -----------------------------
GRID_PATH = Path("data/processed_esa_ee_osm/nuremberg_grid_250m_wgs84.geojson")
OSM_FEATURES_PATH = Path("data/processed_esa_ee_osm/nuremberg_2021_osm_features_250m.csv")

ROADS_PATH = Path("data/osm/2021/gis_osm_roads_free_1.shp")
BUILDINGS_PATH = Path("data/osm/2021/gis_osm_buildings_a_free_1.shp")

# -----------------------------
# 1. LOAD GRID + OSM FEATURES
# -----------------------------
grid = gpd.read_file(GRID_PATH)
osm = pd.read_csv(OSM_FEATURES_PATH)

grid = grid.merge(osm, on="cell_id", how="left")

# project to metric CRS for clean plotting/overlay
grid = grid.to_crs(32632)

# -----------------------------
# 2. LOAD RAW OSM LAYERS
# -----------------------------
roads = gpd.read_file(ROADS_PATH).to_crs(32632)
buildings = gpd.read_file(BUILDINGS_PATH).to_crs(32632)

roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])].copy()
buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

# -----------------------------
# 3. QUICK CHOROPLETHS
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

grid.plot(
    column="road_density",
    cmap="viridis",
    linewidth=0,
    legend=True,
    ax=axes[0]
)
axes[0].set_title("Road Density by 250m Grid Cell")
axes[0].set_axis_off()

grid.plot(
    column="building_area_ratio",
    cmap="magma",
    linewidth=0,
    legend=True,
    ax=axes[1]
)
axes[1].set_title("Building Area Ratio by 250m Grid Cell")
axes[1].set_axis_off()

plt.tight_layout()
plt.show()

# -----------------------------
# 4. STRONGER SANITY CHECK:
#    overlay roads on road_density
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 10))

grid.plot(
    column="road_density",
    cmap="viridis",
    linewidth=0,
    legend=True,
    ax=ax
)
roads.plot(ax=ax, color="black", linewidth=0.3, alpha=0.5)

ax.set_title("Road Density with Raw Road Overlay")
ax.set_axis_off()
plt.tight_layout()
plt.show()

# -----------------------------
# 5. STRONGER SANITY CHECK:
#    overlay buildings on building ratio
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 10))

grid.plot(
    column="building_area_ratio",
    cmap="magma",
    linewidth=0,
    legend=True,
    ax=ax
)
buildings.plot(ax=ax, color="cyan", linewidth=0, alpha=0.15)

ax.set_title("Building Area Ratio with Raw Building Overlay")
ax.set_axis_off()
plt.tight_layout()
plt.show()