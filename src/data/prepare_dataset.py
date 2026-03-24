from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd

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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def prepare_dataset_for_years(
    years: list[int],
    grid: gpd.GeoDataFrame,
    osm_features: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Build feature and label tables for multiple years.

    Args:
        years: List of years to process (e.g., [2016, 2017, 2018])
        grid: Grid GeoDataFrame with cell_id and geometry
        osm_features: Optional pre-computed OSM features to merge

    Returns:
        Dictionary with keys like "dataset_2016", "dataset_2017", etc.
    """
    yearly_datasets: dict[str, pd.DataFrame] = {}

    for year in years:
        logger.info(f"\nProcessing year {year}...")

        # Discover data files
        try:
            sentinel_path = discover_sentinel_safe(year)
            worldcover_path = discover_worldcover_path(year)
        except FileNotFoundError as e:
            logger.error(f"Failed to find data for {year}: {e}")
            logger.error(f"Skipping year {year}")
            continue

        # Build feature and label tables
        logger.info(f"  Building features for {year}...")
        features = build_feature_table(grid, sentinel_path, str(year))

        logger.info(f"  Building labels for {year}...")
        labels = build_label_table(grid, worldcover_path, str(year))

        # Merge features and labels
        logger.info(f"  Merging features and labels for {year}...")
        dataset = features.merge(labels, on=["cell_id", "year_label"], how="left")

        # Add OSM context if provided
        if osm_features is not None:
            dataset = dataset.merge(osm_features, on="cell_id", how="left")

        yearly_datasets[f"dataset_{year}"] = dataset

        # Save individual year dataset
        output_path = PROCESSED_DIR / f"dataset_{year}.csv"
        save_dataframe(dataset, output_path)
        logger.info(f"  Saved dataset to {output_path}")

    return yearly_datasets


def create_change_datasets(yearly_datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Create change datasets for sequential year pairs.

    Args:
        yearly_datasets: Dictionary of yearly datasets from prepare_dataset_for_years

    Returns:
        Dictionary with change datasets (e.g., "change_dataset_2016_2017")
    """
    change_datasets: dict[str, pd.DataFrame] = {}

    # Extract years from keys and sort
    years = sorted(
        [int(key.split("_")[1]) for key in yearly_datasets.keys()]
    )

    # Create change datasets for consecutive years
    for i in range(len(years) - 1):
        year1, year2 = years[i], years[i + 1]

        logger.info(f"\nCreating change dataset for {year1} → {year2}...")

        dataset_key1 = f"dataset_{year1}"
        dataset_key2 = f"dataset_{year2}"

        if dataset_key1 not in yearly_datasets or dataset_key2 not in yearly_datasets:
            logger.warning(f"Missing data for {year1} or {year2}, skipping change dataset")
            continue

        change_dataset = build_change_dataset(
            yearly_datasets[dataset_key1],
            yearly_datasets[dataset_key2],
        )

        change_key = f"change_dataset_{year1}_{year2}"
        change_datasets[change_key] = change_dataset

        # Save change dataset
        output_path = PROCESSED_DIR / f"change_dataset_{year1}_{year2}.csv"
        save_dataframe(change_dataset, output_path)
        logger.info(f"  Saved change dataset to {output_path}")

    return change_datasets


def main(years: list[int] | None = None) -> None:
    """
    Prepare datasets for multiple years with change detection.

    Args:
        years: List of years to process. If None, defaults to [2016, 2017, 2018].
               Can also be set via YEARS_TO_PROCESS environment variable (comma-separated).
    """
    if years is None:
        # Check environment variable first
        years_env = os.getenv("YEARS_TO_PROCESS")
        if years_env:
            years = [int(y.strip()) for y in years_env.split(",")]
        else:
            # Default to 2016, 2017, 2018
            years = [2016, 2017, 2018]

    logger.info(f"\n{'='*70}")
    logger.info(" DATASET PREPARATION FOR MULTIPLE YEARS")
    logger.info(f"{'='*70}")
    logger.info(f"Years to process: {years}")

    ensure_directories([INTERIM_DIR, PROCESSED_DIR])

    # Create grid (once for all years)
    logger.info("\nCreating grid...")
    boundary = load_boundary()
    grid = create_grid(boundary, GRID_SIZE_METERS)
    centroids = grid.geometry.centroid
    grid["centroid_x"] = centroids.x
    grid["centroid_y"] = centroids.y
    save_geodataframe(grid, INTERIM_DIR / "grid.geojson")
    logger.info(f"  Saved grid to {INTERIM_DIR / 'grid.geojson'}")

    # Load OSM features (year-independent)
    osm_features = None
    osm_context_path = discover_osm_context()
    if osm_context_path is not None:
        logger.info("\nBuilding OSM context features...")
        osm_features = build_osm_context_features(grid, osm_context_path)
        logger.info(f"  OSM features: {len(osm_features)} cells")

    # Prepare datasets for all years
    logger.info(f"\n{'='*70}")
    logger.info(" PROCESSING INDIVIDUAL YEARS")
    logger.info(f"{'='*70}")

    yearly_datasets = prepare_dataset_for_years(years, grid, osm_features)

    if not yearly_datasets:
        logger.error("No datasets were successfully created. Exiting.")
        return

    # Create change datasets
    logger.info(f"\n{'='*70}")
    logger.info(" CREATING CHANGE DATASETS")
    logger.info(f"{'='*70}")

    change_datasets = create_change_datasets(yearly_datasets)

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(" SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Successfully created datasets:")
    logger.info(f"  Individual year datasets: {len(yearly_datasets)}")
    for key in sorted(yearly_datasets.keys()):
        logger.info(f"    ✓ {key}")
    logger.info(f"  Change datasets: {len(change_datasets)}")
    for key in sorted(change_datasets.keys()):
        logger.info(f"    ✓ {key}")
    logger.info(f"\nAll outputs saved to: {PROCESSED_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare datasets for multiple years with change detection"
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        help="Years to process (e.g., --years 2016 2017 2018). "
        "Default: 2016 2017 2018. Can also use YEARS_TO_PROCESS env var.",
    )

    args = parser.parse_args()
    main(years=args.years)
