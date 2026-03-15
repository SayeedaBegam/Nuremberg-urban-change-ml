from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling


BAND_PATH_PATTERNS = {
    "b2": ("*B02_10m.jp2", 10),
    "b3": ("*B03_10m.jp2", 10),
    "b4": ("*B04_10m.jp2", 10),
    "b8": ("*B08_10m.jp2", 10),
    "b11": ("*B11_20m.jp2", 20),
    "b12": ("*B12_20m.jp2", 20),
}


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    denominator = np.where(np.abs(denominator) < 1e-6, np.nan, denominator)
    return numerator / denominator


def discover_sentinel_bands(safe_path) -> dict[str, str]:
    """Find the required Sentinel-2 band files inside a SAFE product."""
    safe_path = Path(safe_path).resolve()
    band_paths: dict[str, str] = {}
    for band_name, (pattern, _) in BAND_PATH_PATTERNS.items():
        matches = list(safe_path.rglob(pattern))
        if not matches:
            raise FileNotFoundError(f"Could not find {band_name} in {safe_path}.")
        band_paths[band_name] = str(matches[0])
    return band_paths


def _prepare_datasets(
    band_paths: dict[str, str],
) -> tuple[dict[str, rasterio.io.DatasetReader], ExitStack]:
    stack = ExitStack()
    template = stack.enter_context(rasterio.open(band_paths["b2"]))
    datasets: dict[str, rasterio.io.DatasetReader] = {}
    for band_name, (_, resolution) in BAND_PATH_PATTERNS.items():
        dataset = stack.enter_context(rasterio.open(band_paths[band_name]))
        if resolution == 10:
            datasets[band_name] = dataset
        else:
            datasets[band_name] = stack.enter_context(
                WarpedVRT(
                    dataset,
                    crs=template.crs,
                    transform=template.transform,
                    width=template.width,
                    height=template.height,
                    resampling=Resampling.bilinear,
                )
            )
    return datasets, stack


def _masked_values(dataset: rasterio.io.DatasetReader, geometry) -> np.ndarray:
    out_image, _ = mask(dataset, [geometry], crop=True, filled=False)
    return out_image[0].astype("float32")


def _summarize_cell(datasets: dict[str, rasterio.io.DatasetReader], geometry) -> dict[str, float]:
    """Extract band summaries and spectral indices for one grid cell."""
    summaries: dict[str, float] = {}

    masked = {band_name: _masked_values(dataset, geometry) for band_name, dataset in datasets.items()}

    for band_name, band_array in masked.items():
        band_values = band_array.compressed()
        if band_values.size == 0:
            summaries[f"{band_name}_mean"] = np.nan
            summaries[f"{band_name}_std"] = np.nan
        else:
            summaries[f"{band_name}_mean"] = float(np.nanmean(band_values))
            summaries[f"{band_name}_std"] = float(np.nanstd(band_values))

    b3 = masked["b3"].filled(np.nan)
    b4 = masked["b4"].filled(np.nan)
    b8 = masked["b8"].filled(np.nan)
    b11 = masked["b11"].filled(np.nan)

    ndvi = _safe_ratio(b8 - b4, b8 + b4)
    ndbi = _safe_ratio(b11 - b8, b11 + b8)
    ndwi = _safe_ratio(b3 - b8, b3 + b8)
    brightness_stack = np.stack([b3, b4, b8, b11])
    if np.isnan(brightness_stack).all():
        brightness = np.full_like(b3, np.nan, dtype="float32")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            brightness = np.nanmean(brightness_stack, axis=0)

    summaries["ndvi_mean"] = float(np.nanmean(ndvi)) if not np.isnan(ndvi).all() else np.nan
    summaries["ndvi_std"] = float(np.nanstd(ndvi)) if not np.isnan(ndvi).all() else np.nan
    summaries["ndbi_mean"] = float(np.nanmean(ndbi)) if not np.isnan(ndbi).all() else np.nan
    summaries["ndwi_mean"] = float(np.nanmean(ndwi)) if not np.isnan(ndwi).all() else np.nan
    summaries["brightness_mean"] = float(np.nanmean(brightness)) if not np.isnan(brightness).all() else np.nan
    return summaries


def build_feature_table(grid: gpd.GeoDataFrame, safe_path, year_label: str) -> pd.DataFrame:
    """Create one tabular feature row per grid cell for a given Sentinel-2 SAFE product."""
    rows = []
    band_paths = discover_sentinel_bands(safe_path)
    datasets, stack = _prepare_datasets(band_paths)
    with stack:
        grid_local = grid.to_crs(datasets["b2"].crs)
        for record in grid_local.itertuples(index=False):
            feature_row = {
                "cell_id": record.cell_id,
                "year_label": year_label,
                "centroid_x": record.centroid_x,
                "centroid_y": record.centroid_y,
            }
            feature_row.update(_summarize_cell(datasets, record.geometry))
            rows.append(feature_row)
    return pd.DataFrame(rows)
