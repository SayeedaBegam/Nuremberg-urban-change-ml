import geopandas as gpd
from shapely.geometry import box

from src.utils.config import NUREMBERG_BBOX_4326, RAW_PATHS


def load_boundary() -> gpd.GeoDataFrame:
    """Load the Nuremberg boundary or fall back to a study-area bounding box."""
    if RAW_PATHS["boundary"].exists():
        boundary = gpd.read_file(RAW_PATHS["boundary"])
        if boundary.crs is None:
            raise ValueError("Boundary file is missing a CRS.")
        return boundary.to_crs(3857)

    min_x, min_y, max_x, max_y = NUREMBERG_BBOX_4326
    boundary = gpd.GeoDataFrame(geometry=[box(min_x, min_y, max_x, max_y)], crs=4326)
    return boundary.to_crs(3857)
