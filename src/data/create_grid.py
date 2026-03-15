import geopandas as gpd
import numpy as np
from shapely.geometry import box


def create_grid(boundary: gpd.GeoDataFrame, grid_size: int) -> gpd.GeoDataFrame:
    """Create a square grid clipped to the study boundary."""
    min_x, min_y, max_x, max_y = boundary.total_bounds
    x_coords = np.arange(min_x, max_x, grid_size)
    y_coords = np.arange(min_y, max_y, grid_size)

    cells = [box(x, y, x + grid_size, y + grid_size) for x in x_coords for y in y_coords]
    grid = gpd.GeoDataFrame({"geometry": cells}, crs=boundary.crs)
    clipped = gpd.overlay(grid, boundary[["geometry"]], how="intersection")
    clipped = clipped.reset_index(drop=True)
    clipped["cell_id"] = [f"cell_{idx:05d}" for idx in range(len(clipped))]
    return clipped[["cell_id", "geometry"]]
