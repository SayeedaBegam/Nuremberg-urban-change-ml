import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


RAW_PATHS = {
    "boundary": RAW_DIR / "nuremberg_boundary.geojson",
    "sentinel_t1": RAW_DIR / "sentinel_t1.tif",
    "sentinel_t2": RAW_DIR / "sentinel_t2.tif",
    "worldcover_t1": RAW_DIR / "worldcover_t1.tif",
    "worldcover_t2": RAW_DIR / "worldcover_t2.tif",
    "osm_context": RAW_DIR / "osm_context.gpkg",
}


GRID_SIZE_METERS = int(os.getenv("GRID_SIZE_METERS", "250"))
RANDOM_STATE = 42
NUREMBERG_BBOX_4326 = (10.97, 49.38, 11.15, 49.51)


WORLD_COVER_MAPPING = {
    10: "vegetation",
    20: "vegetation",
    30: "vegetation",
    40: "vegetation",
    50: "built_up",
    60: "other",
    80: "water",
    90: "other",
    95: "other",
    100: "other",
    70: "other",
}


TARGET_COLUMNS = [
    "built_up_prop",
    "vegetation_prop",
    "water_prop",
    "other_prop",
]


CHANGE_TARGET_COLUMNS = [
    "delta_built_up",
    "delta_vegetation",
    "delta_water",
    "delta_other",
]


def discover_worldcover_path(year: int) -> Path:
    explicit = RAW_PATHS[f"worldcover_t{1 if year == 2020 else 2}"]
    if explicit.exists():
        return explicit

    matches = sorted(RAW_DIR.glob(f"ESA_WorldCover_10m_{year}_*_Map.tif"))
    if not matches:
        raise FileNotFoundError(f"WorldCover raster for {year} not found in {RAW_DIR}.")
    return matches[0]


def discover_sentinel_safe(year: int) -> Path:
    explicit = RAW_PATHS[f"sentinel_t{1 if year == 2020 else 2}"]
    if explicit.exists():
        return explicit

    matches = sorted(RAW_DIR.glob(f"S2*_MSIL2A_{year}*.SAFE"))
    if matches:
        return matches[0]

    nested_matches = sorted(RAW_DIR.rglob(f"S2*_MSIL2A_{year}*.SAFE"))
    if not nested_matches:
        raise FileNotFoundError(f"Sentinel-2 SAFE product for {year} not found in {RAW_DIR}.")
    return nested_matches[0]


def discover_osm_context() -> Path | None:
    if RAW_PATHS["osm_context"].exists():
        return RAW_PATHS["osm_context"]

    candidates = sorted(RAW_DIR.glob("*.gpkg")) + sorted(RAW_DIR.glob("*.shp"))
    return candidates[0] if candidates else None
