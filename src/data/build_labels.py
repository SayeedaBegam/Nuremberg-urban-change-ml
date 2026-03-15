from __future__ import annotations

from collections import Counter

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask

from src.utils.config import TARGET_COLUMNS, WORLD_COVER_MAPPING


GROUP_ORDER = ["built_up", "vegetation", "water", "other"]


def _map_group(value: int) -> str:
    return WORLD_COVER_MAPPING.get(int(value), "other")


def _label_props(dataset: rasterio.io.DatasetReader, geometry) -> dict[str, float]:
    """Compute grouped land-cover proportions inside a cell."""
    out_image, _ = mask(dataset, [geometry], crop=True, filled=False)
    values = out_image[0].compressed()
    counts = Counter(_map_group(value) for value in values if value > 0)
    total = max(sum(counts.values()), 1)
    return {f"{group}_prop": counts.get(group, 0) / total for group in GROUP_ORDER}


def build_label_table(grid: gpd.GeoDataFrame, raster_path, year_label: str) -> pd.DataFrame:
    rows = []
    with rasterio.open(raster_path) as dataset:
        grid_local = grid.to_crs(dataset.crs)
        for record in grid_local.itertuples(index=False):
            row = {"cell_id": record.cell_id, "year_label": year_label}
            row.update(_label_props(dataset, record.geometry))
            rows.append(row)

    labels = pd.DataFrame(rows)
    for target in TARGET_COLUMNS:
        if target not in labels:
            labels[target] = np.nan
    return labels
