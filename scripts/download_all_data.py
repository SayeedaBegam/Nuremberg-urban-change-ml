#!/usr/bin/env python
"""Orchestrate downloading of Sentinel-2 and WorldCover data for multiple years."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_command(cmd: list[str], description: str) -> bool:
    """Run a shell command and return success status."""
    logger.info(f"\n{'='*70}")
    logger.info(f" {description}")
    logger.info(f"{'='*70}")
    logger.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    if result.returncode == 0:
        logger.info(f"✓ {description} completed successfully")
        return True
    else:
        logger.error(f"✗ {description} failed with return code {result.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download all required satellite data (Sentinel-2 + WorldCover)"
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
        help="Month for Sentinel-2 (default: 6 for June)",
    )
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=50,
        help="Maximum cloud cover % for Sentinel-2 (default: 50)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "raw",
        help="Data directory for downloads",
    )
    parser.add_argument(
        "--skip-sentinel",
        action="store_true",
        help="Skip Sentinel-2 download",
    )
    parser.add_argument(
        "--skip-worldcover",
        action="store_true",
        help="Skip WorldCover download",
    )

    args = parser.parse_args()

    logger.info(f"\n{'='*70}")
    logger.info(" SATELLITE DATA DOWNLOAD ORCHESTRATION")
    logger.info(f"{'='*70}")
    logger.info(f"Years: {args.years}")
    logger.info(f"Sentinel-2 month: {args.month}")
    logger.info(f"Max cloud cover: {args.max_cloud}%")
    logger.info(f"Data directory: {args.data_dir}")

    # Ensure data directory exists
    args.data_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Download Sentinel-2
    if not args.skip_sentinel:
        cmd = [
            sys.executable,
            "-m",
            "scripts.download_sentinel2",
            "--years",
            *[str(year) for year in args.years],
            "--month",
            str(args.month),
            "--max-cloud",
            str(args.max_cloud),
            "--output-dir",
            str(args.data_dir),
        ]
        results["Sentinel-2"] = run_command(cmd, "Download Sentinel-2 L2A imagery")
    else:
        logger.info("Skipping Sentinel-2 download")

    # Download WorldCover
    if not args.skip_worldcover:
        cmd = [
            sys.executable,
            "-m",
            "scripts.download_worldcover",
            "--years",
            *[str(year) for year in args.years],
            "--output-dir",
            str(args.data_dir),
        ]
        results["WorldCover"] = run_command(cmd, "Download ESA WorldCover data")
    else:
        logger.info("Skipping WorldCover download")

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(" DOWNLOAD SUMMARY")
    logger.info(f"{'='*70}")

    all_successful = all(results.values())

    for source, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{source:20} {status}")

    logger.info(f"{'='*70}")

    if all_successful:
        logger.info("All downloads completed successfully!")
        logger.info(f"Data saved to: {args.data_dir}")
        return 0
    else:
        logger.error("Some downloads failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
