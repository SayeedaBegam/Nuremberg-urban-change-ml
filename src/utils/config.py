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


def discover_all_sentinel_years() -> dict[int, Path]:
    """
    Discover all available Sentinel-2 SAFE products in RAW_DIR.

    Returns:
        Dictionary mapping year (int) to SAFE directory path
    """
    years_map: dict[int, Path] = {}

    # Search for SAFE directories with year in name
    # Pattern: S2*_MSIL2A_YYYYMMDD*.SAFE
    safe_dirs = sorted(RAW_DIR.glob("S2*_MSIL2A_*.SAFE"))

    for safe_dir in safe_dirs:
        # Extract year from directory name: S2A_MSIL2A_20200615T102031_N0214_R065_T32UQD_20200615T102026.SAFE
        # The acquisition date YYYYMMDDTHHMMSS is at index 2
        parts = safe_dir.name.split("_")
        year = None

        try:
            # The acquisition date is at parts[2] in format YYYYMMDDTHHMMSS
            if len(parts) >= 3:
                date_str = parts[2]
                if len(date_str) >= 4 and date_str[:4].isdigit():
                    year = int(date_str[:4])
        except (IndexError, ValueError):
            pass

        if year is not None and year not in years_map:
            years_map[year] = safe_dir

    return years_map


def discover_all_worldcover_years() -> dict[int, Path]:
    """
    Discover all available WorldCover rasters in RAW_DIR.

    Returns:
        Dictionary mapping year (int) to GeoTIFF path
    """
    years_map: dict[int, Path] = {}

    # Map symbolic names to years (backward compatibility)
    symbolic_mapping = {
        "worldcover_t1.tif": 2020,
        "worldcover_t2.tif": 2021,
    }

    # First check for symbolic names
    for filename, year in symbolic_mapping.items():
        path = RAW_DIR / filename
        if path.exists() and year not in years_map:
            years_map[year] = path

    # Search for ESA WorldCover GeoTIFFs with year in filename
    # Pattern: ESA_WorldCover_10m_YYYY_*.tif
    worldcover_files = sorted(RAW_DIR.glob("ESA_WorldCover_10m_*_*.tif"))

    for wc_file in worldcover_files:
        # Extract year from filename: ESA_WorldCover_10m_YYYY_*_Map.tif
        parts = wc_file.stem.split("_")
        year = None

        try:
            # Year is at position 3 for ESA_WorldCover_10m_YYYY_*
            if len(parts) >= 4 and parts[3].isdigit():
                year = int(parts[3])
        except (IndexError, ValueError):
            pass

        if year is not None and year not in years_map:
            years_map[year] = wc_file

    return years_map
