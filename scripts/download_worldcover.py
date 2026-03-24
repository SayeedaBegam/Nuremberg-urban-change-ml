#!/usr/bin/env python
"""Download ESA WorldCover raster data for Nuremberg for specified years.

NOTE: ESA WorldCover data requires manual download or use of specialized tools.
Visit: https://esa-worldcover.org/ to download files directly.

For 2016-2018 data, you can:
1. Download from ESA website (https://esa-worldcover.org/maps/)
2. Use Google Earth Engine (requires account)
3. Use rasterio COG remote reading if available
4. Generate mock/synthetic data for testing (see create_mock_data.py)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.mask import mask

from src.data.load_boundary import load_boundary
from src.utils.config import RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ESA WorldCover availability
WORLDCOVER_INFO = {
    2015: "Available in ESA v100",
    2016: "Available in ESA v100 & v200",
    2017: "Available in ESA v100 & v200",
    2018: "Available in ESA v100 & v200",
    2019: "Available in ESA v100 & v200",
    2020: "Available in ESA v200 (latest)",
}


def clip_worldcover(
    input_raster: Path,
    boundary: gpd.GeoDataFrame,
    output_raster: Path,
) -> None:
    """
    Clip WorldCover raster to study area boundary.

    Args:
        input_raster: Path to input GeoTIFF
        boundary: GeoDataFrame with clipping geometry
        output_raster: Path to output clipped GeoTIFF
    """
    # Ensure boundary is in WGS84 (EPSG:4326)
    if boundary.crs != "EPSG:4326":
        boundary = boundary.to_crs("EPSG:4326")

    # Read input raster
    with rasterio.open(input_raster) as src:
        # Clip using rasterio mask
        clipped_data, clipped_transform = mask(
            src,
            [geom for geom in boundary.geometry],
            crop=True,
            filled=True,
            fill_value=0,
        )

        # Update metadata
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": clipped_data.shape[1],
                "width": clipped_data.shape[2],
                "transform": clipped_transform,
                "compress": "lzw",
            }
        )

        # Write clipped raster
        with rasterio.open(output_raster, "w", **out_meta) as dst:
            dst.write(clipped_data)

    logger.info(f"Clipped raster saved to {output_raster}")


def process_downloaded_worldcover(
    input_path: Path,
    output_name: str,
    boundary: gpd.GeoDataFrame | None = None,
    output_dir: Path = RAW_DIR,
) -> Path | None:
    """
    Process manually downloaded WorldCover file.

    Args:
        input_path: Path to locally downloaded WorldCover GeoTIFF
        output_name: Output filename (e.g., "worldcover_2016.tif")
        boundary: Optional boundary to clip to
        output_dir: Directory to save processed file

    Returns:
        Path to processed file, or None if failed
    """
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return None

    output_path = output_dir / output_name

    try:
        if boundary is not None:
            logger.info(f"Clipping {input_path.name} to boundary...")
            clip_worldcover(input_path, boundary, output_path)
        else:
            logger.info(f"Copying {input_path.name} to {output_path}...")
            import shutil

            shutil.copy2(input_path, output_path)

        logger.info(f"✓ Processed WorldCover saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Process manually downloaded ESA WorldCover data for Nuremberg"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to downloaded WorldCover GeoTIFF file",
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Year of the WorldCover data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Output directory (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Skip clipping to boundary",
    )
    parser.add_argument(
        "--list-available",
        action="store_true",
        help="List available WorldCover years",
    )

    args = parser.parse_args()

    if args.list_available:
        logger.info("ESA WorldCover Availability:")
        logger.info("=" * 70)
        for year, info in WORLDCOVER_INFO.items():
            logger.info(f"  {year}: {info}")
        logger.info("=" * 70)
        logger.info("\nTo download data:")
        logger.info("  1. Visit https://esa-worldcover.org/maps/")
        logger.info("  2. Select your year and region")
        logger.info("  3. Download the GeoTIFF")
        logger.info("  4. Run this script with --input <path> --year <year>")
        return

    if args.input is None or args.year is None:
        logger.error("Please provide --input and --year arguments")
        logger.error("Or use --list-available to see available years")
        return

    # Load boundary for clipping
    boundary = None
    if not args.no_clip:
        logger.info("Loading Nuremberg boundary...")
        try:
            boundary = load_boundary().to_crs("EPSG:4326")
        except Exception as e:
            logger.warning(f"Could not load boundary: {e}. Will not clip.")

    # Process the file
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"ESA_WorldCover_10m_{args.year}_Map.tif"
    result = process_downloaded_worldcover(
        args.input, output_filename, boundary, args.output_dir
    )

    if result:
        logger.info(f"\n✓ Successfully processed WorldCover for {args.year}")
        logger.info(f"  Output: {result}")
    else:
        logger.error(f"✗ Failed to process WorldCover for {args.year}")


if __name__ == "__main__":
    main()

