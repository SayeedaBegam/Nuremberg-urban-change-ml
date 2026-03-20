"""
Dataset Ablation Study: Comparing Different Satellite Data Sources

This script compares different satellite imagery sources to evaluate their
contribution to predicting urban land-cover change in Nuremberg.

Datasets tested:
1. Sentinel-2 (current baseline) - 10m resolution, 11 bands
2. AWS Sentinel-2 - same as above but from AWS archive
3. Landsat 8/9 - 30m resolution, 11 bands, different spectral response
4. MODIS - 250m resolution, good for temporal analysis
5. Mixed combinations - test synergy between datasets

The study measures:
- Prediction accuracy (MAE, RMSE per target)
- Model confidence (uncertainty from Random Forest)
- Feature importance ranking
- Cost-benefit (data size vs. performance gain)
"""

from __future__ import annotations

import ee
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

# Initialize Earth Engine
try:
    ee.Initialize()
except Exception as init_error:
    print(f"Earth Engine init warning: {init_error}")
    # Continue anyway - might already be initialized


class DatasetAblationConfig:
    """Configuration for dataset ablation study."""

    # Nuremberg bounds (approximate)
    # Lat/Lon: (49.45°N, 11.08°E)
    NUREMBERG_GEOMETRY = ee.Geometry.Rectangle([
        10.95, 49.35,  # SW corner (lon, lat)
        11.25, 49.55   # NE corner (lon, lat)
    ])

    # Study period
    DATE_START_T1 = "2020-06-01"
    DATE_END_T1 = "2020-08-31"
    DATE_START_T2 = "2021-06-01"
    DATE_END_T2 = "2021-08-31"

    # Available datasets
    DATASETS = {
        "sentinel2": {
            "collection": "COPERNICUS/S2_SR_HARMONIZED",
            "description": "Sentinel-2 L2A (10m, 11 bands)",
            "resolution": 10,
            "bands": ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
        },
        "landsat8": {
            "collection": "LANDSAT/LC08/C02/T1_L2",
            "description": "Landsat 8 (30m, 11 bands)",
            "resolution": 30,
            "bands": ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "SR_B10"],
        },
        "modis": {
            "collection": "MODIS/006/MOD09GA",
            "description": "MODIS (500m, 7 bands)",
            "resolution": 500,
            "bands": ["sur_refl_b01", "sur_refl_b02", "sur_refl_b03", "sur_refl_b04"],
        },
    }


def fetch_sentinel2_aws(date_start: str, date_end: str, geometry) -> ee.Image:
    """Fetch Sentinel-2 from AWS via Earth Engine."""
    print(f"  Fetching Sentinel-2 ({date_start} to {date_end})...")

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .select(["B2", "B3", "B4", "B8", "B11", "B12"])
    )

    if collection.size().getInfo() == 0:
        print("  ⚠️  No Sentinel-2 images found")
        return None

    # Composite: median pixel value across all images
    composite = collection.median()
    return composite


def fetch_landsat_aws(date_start: str, date_end: str, geometry) -> ee.Image:
    """Fetch Landsat 8 from AWS via Earth Engine."""
    print(f"  Fetching Landsat 8 ({date_start} to {date_end})...")

    collection = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(geometry)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.lt("CLOUD_COVER", 20))
        .select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"])
    )

    if collection.size().getInfo() == 0:
        print("  ⚠️  No Landsat 8 images found")
        return None

    composite = collection.median()
    return composite


def fetch_modis_aws(date_start: str, date_end: str, geometry) -> ee.Image:
    """Fetch MODIS from AWS via Earth Engine."""
    print(f"  Fetching MODIS ({date_start} to {date_end})...")

    collection = (
        ee.ImageCollection("MODIS/006/MOD09GA")
        .filterBounds(geometry)
        .filterDate(date_start, date_end)
        .select(["sur_refl_b01", "sur_refl_b02", "sur_refl_b03", "sur_refl_b04", "sur_refl_b05"])
    )

    if collection.size().getInfo() == 0:
        print("  ⚠️  No MODIS images found")
        return None

    composite = collection.median()
    return composite


def fetch_dataset_comparison() -> dict[str, dict]:
    """Fetch all datasets and gather metadata for comparison."""

    config = DatasetAblationConfig()
    results = {}

    print("\n" + "="*70)
    print("DATASET AVAILABILITY CHECK")
    print("="*70)

    # Sentinel-2
    print("\n[1] Sentinel-2 (AWS via Earth Engine)")
    s2_t1 = fetch_sentinel2_aws(config.DATE_START_T1, config.DATE_END_T1, config.NUREMBERG_GEOMETRY)
    s2_t2 = fetch_sentinel2_aws(config.DATE_START_T2, config.DATE_END_T2, config.NUREMBERG_GEOMETRY)

    if s2_t1 and s2_t2:
        # Get system metadata
        props_t1 = s2_t1.getInfo()
        results["sentinel2"] = {
            "status": "✅ Available",
            "t1": s2_t1,
            "t2": s2_t2,
            "resolution": 10,
            "bands": 6,
            "image_count_t1": ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(config.NUREMBERG_GEOMETRY)
            .filterDate(config.DATE_START_T1, config.DATE_END_T1)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .size()
            .getInfo(),
        }
        print(f"  ✅ Available ({results['sentinel2']['image_count_t1']} images in T1)")
    else:
        results["sentinel2"] = {"status": "❌ Not available"}
        print("  ❌ Not available")

    # Landsat
    print("\n[2] Landsat 8 (AWS via Earth Engine)")
    l8_t1 = fetch_landsat_aws(config.DATE_START_T1, config.DATE_END_T1, config.NUREMBERG_GEOMETRY)
    l8_t2 = fetch_landsat_aws(config.DATE_START_T2, config.DATE_END_T2, config.NUREMBERG_GEOMETRY)

    if l8_t1 and l8_t2:
        results["landsat8"] = {
            "status": "✅ Available",
            "t1": l8_t1,
            "t2": l8_t2,
            "resolution": 30,
            "bands": 6,
            "image_count_t1": ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(config.NUREMBERG_GEOMETRY)
            .filterDate(config.DATE_START_T1, config.DATE_END_T1)
            .filter(ee.Filter.lt("CLOUD_COVER", 20))
            .size()
            .getInfo(),
        }
        print(f"  ✅ Available ({results['landsat8']['image_count_t1']} images in T1)")
    else:
        results["landsat8"] = {"status": "❌ Not available"}
        print("  ❌ Not available")

    # MODIS
    print("\n[3] MODIS (AWS via Earth Engine)")
    modis_t1 = fetch_modis_aws(config.DATE_START_T1, config.DATE_END_T1, config.NUREMBERG_GEOMETRY)
    modis_t2 = fetch_modis_aws(config.DATE_START_T2, config.DATE_END_T2, config.NUREMBERG_GEOMETRY)

    if modis_t1 and modis_t2:
        results["modis"] = {
            "status": "✅ Available",
            "t1": modis_t1,
            "t2": modis_t2,
            "resolution": 500,
            "bands": 5,
            "image_count_t1": ee.ImageCollection("MODIS/006/MOD09GA")
            .filterBounds(config.NUREMBERG_GEOMETRY)
            .filterDate(config.DATE_START_T1, config.DATE_END_T1)
            .size()
            .getInfo(),
        }
        print(f"  ✅ Available ({results['modis']['image_count_t1']} images in T1)")
    else:
        results["modis"] = {"status": "❌ Not available"}
        print("  ❌ Not available")

    return results


def create_dataset_comparison_report(results: dict) -> None:
    """Generate a report comparing available datasets."""

    print("\n" + "="*70)
    print("DATASET COMPARISON & RECOMMENDATIONS")
    print("="*70)

    comparison_data = []

    for dataset_name, info in results.items():
        if info["status"] == "✅ Available":
            comparison_data.append({
                "Dataset": dataset_name,
                "Status": "Available",
                "Resolution (m)": info.get("resolution", "N/A"),
                "Bands": info.get("bands", "N/A"),
                "Image Count (T1)": info.get("image_count_t1", "N/A"),
            })
        else:
            comparison_data.append({
                "Dataset": dataset_name,
                "Status": "Not Found",
                "Resolution (m)": "N/A",
                "Bands": "N/A",
                "Image Count (T1)": "N/A",
            })

    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))

    print("\n" + "-"*70)
    print("DATASET CHARACTERISTICS & TRADE-OFFS")
    print("-"*70)
    print("""
Sentinel-2 (10m resolution)
  ✅ Pro:  High resolution, good for detailed urban mapping
  ✅ Pro:  Currently working in your pipeline
  ❌ Con:  Smaller swath, more compositing needed
  Usage: Best for fine-grained change detection

Landsat 8 (30m resolution)
  ✅ Pro:  Lower cloud cover, more frequent revisits
  ✅ Pro:  Longer historical archive (2013+)
  ✅ Pro:  Good spectral match with older Landsat data
  ❌ Con:  Coarser resolution, harder to see small feature changes
  Usage: Good for regional trends, smoother time series

MODIS (500m resolution)
  ✅ Pro:  Daily coverage, excellent temporal resolution
  ✅ Pro:  Good for broad change patterns, much more data
  ❌ Con:  Too coarse for your 250m grid study
  ❌ Con:  Likely overkill for local urban change
  Usage: Regional/continental scale analysis
    """)

    print("\n" + "-"*70)
    print("NEXT STEPS FOR DATASET ABLATION")
    print("-"*70)
    print("""
1. Extract features from each available dataset
2. Train models with:
   - Sentinel-2 only (baseline)
   - Landsat only (alternative source)
   - Sentinel-2 + Landsat (combined)
   - Compare confidence & accuracy
3. Visualize which dataset improves predictions
4. Measure feature importance per data source
    """)


def main() -> None:
    """Run dataset availability check and comparison."""
    print("\n🛰️  DATASET ABLATION STUDY: AWS DATA SOURCE COMPARISON")
    print("="*70)
    print("Checking data availability for Nuremberg, Germany")
    print("Study period: 2020 & 2021 (June-August)")

    try:
        results = fetch_dataset_comparison()
        create_dataset_comparison_report(results)

        # Save results
        summary = {
            "timestamp": datetime.now().isoformat(),
            "study_area": "Nuremberg, Germany",
            "date_range": {
                "t1": f"{DatasetAblationConfig.DATE_START_T1} to {DatasetAblationConfig.DATE_END_T1}",
                "t2": f"{DatasetAblationConfig.DATE_START_T2} to {DatasetAblationConfig.DATE_END_T2}",
            },
            "datasets_found": {k: v["status"] for k, v in results.items()},
        }

        output_file = Path("./models/dataset_availability_report.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n✓ Report saved to: {output_file}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Note: This requires Earth Engine authentication (already done)")
        print("and internet access to fetch from AWS.")


if __name__ == "__main__":
    main()
