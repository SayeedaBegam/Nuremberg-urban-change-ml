import geopandas as gpd

gdf = gpd.read_file("data/raw/nuremberg_boundary.geojson")

# keep only polygon geometry
gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

# keep only geometry column
gdf = gdf[["geometry"]]

# fix geometry
gdf["geometry"] = gdf["geometry"].buffer(0)

# save as shapefile
gdf.to_file("data/boundary_2/nuremberg_boundary.shp")