from __future__ import annotations

from src.data.build_change_dataset import build_change_dataset
from src.data.build_features import build_feature_table
from src.data.build_labels import build_label_table
from src.data.build_osm_features import build_osm_context_features
from src.data.create_grid import create_grid
from src.data.load_boundary import load_boundary
from src.utils.config import (
    GRID_SIZE_METERS,
    INTERIM_DIR,
    PROCESSED_DIR,
    discover_osm_context,
    discover_sentinel_safe,
    discover_worldcover_path,
)
from src.utils.io import ensure_directories, save_dataframe, save_geodataframe


def main() -> None:
    ensure_directories([INTERIM_DIR, PROCESSED_DIR])
    sentinel_t1_path = discover_sentinel_safe(2020)
    sentinel_t2_path = discover_sentinel_safe(2021)
    worldcover_t1_path = discover_worldcover_path(2020)
    worldcover_t2_path = discover_worldcover_path(2021)

    boundary = load_boundary()
    grid = create_grid(boundary, GRID_SIZE_METERS)
    centroids = grid.geometry.centroid
    grid["centroid_x"] = centroids.x
    grid["centroid_y"] = centroids.y
    save_geodataframe(grid, INTERIM_DIR / "grid.geojson")

    features_t1 = build_feature_table(grid, sentinel_t1_path, "t1")
    features_t2 = build_feature_table(grid, sentinel_t2_path, "t2")
    labels_t1 = build_label_table(grid, worldcover_t1_path, "t1")
    labels_t2 = build_label_table(grid, worldcover_t2_path, "t2")

    dataset_t1 = features_t1.merge(labels_t1, on=["cell_id", "year_label"], how="left")
    dataset_t2 = features_t2.merge(labels_t2, on=["cell_id", "year_label"], how="left")

    osm_context_path = discover_osm_context()
    if osm_context_path is not None:
        osm_features = build_osm_context_features(grid, osm_context_path)
        dataset_t1 = dataset_t1.merge(osm_features, on="cell_id", how="left")
        dataset_t2 = dataset_t2.merge(osm_features, on="cell_id", how="left")

    change_dataset = build_change_dataset(dataset_t1, dataset_t2)

    save_dataframe(dataset_t1, PROCESSED_DIR / "dataset_t1.csv")
    save_dataframe(dataset_t2, PROCESSED_DIR / "dataset_t2.csv")
    save_dataframe(change_dataset, PROCESSED_DIR / "change_dataset.csv")


if __name__ == "__main__":
    main()
