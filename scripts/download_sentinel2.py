#!/usr/bin/env python
"""Download Sentinel-2 L2A imagery for Nuremberg for specified years."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import geopandas as gpd
from sentinelsat import SentinelAPI

from src.data.load_boundary import load_boundary
from src.utils.config import RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def download_sentinel2_for_year(
    api: SentinelAPI,
    boundary: gpd.GeoDataFrame,
    year: int,
    month: int = 6,
    max_cloud_percent: float = 50,
    output_dir: Path = RAW_DIR,
) -> dict[str, str]:
    """
    Download Sentinel-2 L2A imagery for a given year and month.

    Args:
        api: SentinelAPI instance
        boundary: GeoDataFrame with study area geometry (CRS must be 4326)
        year: Year to download (e.g., 2016, 2017, 2018)
        month: Month to download (default June to match existing 2020/2021 data)
        max_cloud_percent: Maximum cloud cover percentage to accept
        output_dir: Directory to save downloaded data

    Returns:
        Dictionary mapping product ID to local file path
    """
    # Convert boundary to WGS84 if needed
    if boundary.crs != "EPSG:4326":
        boundary = boundary.to_crs("EPSG:4326")

    geom = boundary.iloc[0].geometry.__geo_interface__

    # Define search date range (entire month)
    # sentinelsat expects YYYYMMDD format (without hyphens)
    start_date = f"{year}{month:02d}01"
    end_date = f"{year}{month:02d}{28 if month == 2 else 30 if month in [4, 6, 9, 11] else 31}"

    logger.info(f"Searching for Sentinel-2 L2A products for {start_date} to {end_date}...")
    logger.info(f"Max cloud cover: {max_cloud_percent}%")

    # Search for Sentinel-2 L2A products
    products = api.query(
        geometry=geom,
        date=(start_date, end_date),
        platformname="Sentinel-2",
        producttype="S2MSL2A",  # L2A products
        cloudcoverpercentage=(0, max_cloud_percent),
    )

    if not products:
        logger.warning(f"No Sentinel-2 products found for {year}-{month:02d}")
        return {}

    # Sort by cloud cover and take the best one
    products_df = api.to_dataframe(products).sort_values("cloudcoverpercentage")
    best_product = products_df.iloc[0]

    logger.info(
        f"Found {len(products)} products. Downloading best: "
        f"{best_product.name} (cloud cover: {best_product['cloudcoverpercentage']}%)"
    )

    # Download the SAFE product
    try:
        api.download(best_product.name, directory_path=str(output_dir), checksum=True)
        logger.info(f"Successfully downloaded {best_product.name} to {output_dir}")
        return {best_product.name: str(output_dir / best_product.name)}
    except Exception as e:
        logger.error(f"Failed to download {best_product.name}: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 L2A products for Nuremberg"
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2016, 2017, 2018],
        help="Years to download (default: 2016 2017 2018)",
    )
    parser.add_argument(
        "--month",
        type=int,
        default=6,
        help="Month to download (default: 6 for June, matching existing data)",
    )
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=50,
        help="Maximum cloud cover percentage (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Output directory (default: {RAW_DIR})",
    )

    args = parser.parse_args()

    # Load boundary
    logger.info("Loading Nuremberg boundary...")
    boundary = load_boundary()  # Returns EPSG:3857
    boundary = boundary.to_crs("EPSG:4326")  # Convert to WGS84 for sentinelsat

    # Initialize Copernicus Hub API (anonymous access)
    logger.info("Connecting to Copernicus Hub (anonymous)...")
    api = SentinelAPI(None, None, api_url="https://apihub.copernicus.eu/apihub")

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Download data for each year
    all_downloads = {}
    for year in sorted(args.years):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing year {year}")
        logger.info(f"{'='*60}")

        downloads = download_sentinel2_for_year(
            api,
            boundary,
            year,
            month=args.month,
            max_cloud_percent=args.max_cloud,
            output_dir=args.output_dir,
        )
        all_downloads.update(downloads)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Download Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total products downloaded: {len(all_downloads)}")
    for product_id, local_path in all_downloads.items():
        logger.info(f"  ✓ {product_id}")
    logger.info(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
