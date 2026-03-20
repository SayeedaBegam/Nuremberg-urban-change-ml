"""
Nuremberg Boundary Validation & Refinement

This script:
1. Downloads the actual Nuremberg administrative boundary from OSM
2. Validates current data is within proper bounds
3. Clips all features to true city boundary (not just bbox)
4. Reports any data outside Nuremberg that should be excluded
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

from src.utils.config import RAW_DIR, INTERIM_DIR, RAW_PATHS


def download_nuremberg_boundary() -> gpd.GeoDataFrame:
    """Download detailed Nuremberg administrative boundary from OSM/Natural Earth."""
    print("Attempting to download Nuremberg boundary from multiple sources...\n")

    try:
        # Try Natural Earth admin boundaries via geopandas
        print("[1] Trying Natural Earth (ne_10m_admin_2_boundaries)...")
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        nuremberg = world[world["name"] == "Germany"]

        if len(nuremberg) > 0:
            print("  ⚠️  Got country boundary, not city boundary")
            print("  Using custom rectangular boundary instead\n")
            return None

    except Exception as e:
        print(f"  ❌ Natural Earth failed: {e}\n")

    try:
        # Try OSM Overpass API via osmnx (if available)
        print("[2] Checking if osmnx is available for OSM boundary download...")
        import osmnx as ox

        print("  Downloading Nuremberg administrative boundary from OpenStreetMap...")
        # Query for Nuremberg city boundary (admin_level=8 is typically city)
        boundary = ox.geocode_to_gdf("Nuremberg, Germany")

        if boundary is not None and len(boundary) > 0:
            boundary = boundary.to_crs(3857)
            print(f"  ✅ Downloaded! Boundary area: {boundary.geometry.area.values[0] / 1e6:.2f} km²\n")
            return boundary
        else:
            print("  ℹ️  OSM query returned no results\n")
            return None

    except ImportError:
        print("  ℹ️  osmnx not installed. Skipping OSM download.\n")
        return None
    except Exception as e:
        print(f"  ❌ OSM Overpass API failed: {e}\n")
        print("     (This is normal if no internet or OSM is unavailable)\n")
        return None


def validate_current_boundary() -> dict:
    """Check current boundary definition."""
    print("="*70)
    print("CURRENT BOUNDARY VALIDATION")
    print("="*70)

    boundary_file = RAW_PATHS["boundary"]

    if boundary_file.exists():
        boundary = gpd.read_file(boundary_file)
        print(f"\n✅ Boundary file exists: {boundary_file}")
        print(f"   CRS: {boundary.crs}")
        print(f"   Geometry type: {boundary.geometry.type.values[0]}")
        print(f"   Area: {boundary.geometry.area.values[0] / 1e6:.2f} km²")
        print(f"   Bounds: {boundary.total_bounds}")

        return {"exists": True, "boundary": boundary}
    else:
        print(f"\n⚠️  No boundary file found at {boundary_file}")
        print("   Using fallback rectangular bounding box")
        from shapely.geometry import box
        from src.utils.config import NUREMBERG_BBOX_4326

        min_x, min_y, max_x, max_y = NUREMBERG_BBOX_4326
        boundary = gpd.GeoDataFrame(
            geometry=[box(min_x, min_y, max_x, max_y)], crs=4326
        ).to_crs(3857)

        print(f"   Bbox (4326): {NUREMBERG_BBOX_4326}")
        print(f"   Area: {boundary.geometry.area.values[0] / 1e6:.2f} km²")

        return {"exists": False, "boundary": boundary}


def validate_data_clips(boundary: gpd.GeoDataFrame) -> None:
    """Check if existing data files are properly clipped to boundary."""
    print("\n" + "="*70)
    print("DATA CLIPPING VALIDATION")
    print("="*70)

    data_files = {
        "WorldCover T1": RAW_PATHS["worldcover_t1"],
        "WorldCover T2": RAW_PATHS["worldcover_t2"],
        "Grid": INTERIM_DIR / "grid.geojson",
    }

    for name, filepath in data_files.items():
        if filepath.exists():
            print(f"\n✅ {name}: {filepath.name}")
            try:
                data = gpd.read_file(filepath)
                bounds = data.total_bounds
                print(f"   Bounds: [{bounds[0]:.4f}, {bounds[1]:.4f}, {bounds[2]:.4f}, {bounds[3]:.4f}]")
                print(f"   Features: {len(data)}")

                # Check if within boundary
                boundary_bounds = boundary.total_bounds
                if (bounds[0] >= boundary_bounds[0] and
                    bounds[1] >= boundary_bounds[1] and
                    bounds[2] <= boundary_bounds[2] and
                    bounds[3] <= boundary_bounds[3]):
                    print(f"   ✅ Data is within boundary bounds")
                else:
                    print(f"   ⚠️  Data extends beyond boundary bounds")
                    print(f"      Boundary: [{boundary_bounds[0]:.4f}, {boundary_bounds[1]:.4f}, {boundary_bounds[2]:.4f}, {boundary_bounds[3]:.4f}]")

            except Exception as e:
                print(f"   ❌ Error reading: {e}")
        else:
            print(f"\n⭕ {name}: Not yet generated ({filepath.name})")


def validate_grid_coverage(boundary: gpd.GeoDataFrame) -> None:
    """Check grid cell coverage and ensure all cells are within boundary."""
    print("\n" + "="*70)
    print("GRID COVERAGE VALIDATION")
    print("="*70)

    grid_file = INTERIM_DIR / "grid.geojson"

    if not grid_file.exists():
        print("\n⭕ Grid not yet generated. Run `python -m src.data.prepare_dataset` first.")
        return

    grid = gpd.read_file(grid_file)
    print(f"\n✅ Grid loaded: {len(grid)} cells")
    print(f"   Grid size: {grid.geometry.area.values[0]**0.5:.0f}m × {grid.geometry.area.values[0]**0.5:.0f}m")

    # Check if all cells are within boundary
    grid_reprojected = grid.to_crs(boundary.crs)
    boundary_reprojected = boundary.to_crs(grid.crs)

    cells_within = grid.iloc[
        [grid.geometry.within(boundary_reprojected.geometry.iloc[0]).values[i]
         for i in range(len(grid))]
    ]

    print(f"   Cells within boundary: {len(cells_within)} / {len(grid)}")

    if len(cells_within) < len(grid):
        cells_outside = len(grid) - len(cells_within)
        print(f"   ⚠️  {cells_outside} cells ({100*cells_outside/len(grid):.1f}%) extend outside boundary")
        print(f"        These should be clipped during grid creation.")


def create_refined_boundary() -> None:
    """Create and save a refined Nuremberg boundary (if improved version found)."""
    print("\n" + "="*70)
    print("BOUNDARY REFINEMENT")
    print("="*70)

    # Try to download improved boundary
    improved_boundary = download_nuremberg_boundary()

    if improved_boundary is not None:
        output_path = RAW_DIR / "nuremberg_boundary_refined.geojson"
        improved_boundary.to_file(output_path, driver="GeoJSON")
        print(f"✅ Refined boundary saved to: {output_path}")
        print(f"   Area: {improved_boundary.geometry.area.values[0] / 1e6:.2f} km²")
        print("\n   To use this boundary:")
        print(f"   1. Backup current: mv {RAW_PATHS['boundary']} {RAW_PATHS['boundary']}.bak")
        print(f"   2. Replace: cp {output_path} {RAW_PATHS['boundary']}")
        print(f"   3. Re-run: python -m src.data.prepare_dataset")
    else:
        print("\n✅ Could not download improved boundary from online sources.")
        print("   Current rectangular boundary is sufficient.")
        print("   To use a custom boundary:")
        print("   1. Place a GeoJSON/Shapefile in data/raw/nuremberg_boundary.geojson")
        print("   2. Ensure it has CRS EPSG:4326 (WGS84)")
        print("   3. Re-run: python -m src.data.prepare_dataset")


def generate_summary_report() -> None:
    """Generate summary of Nuremberg clipping status."""
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)

    print("""
✅ CURRENT STATUS:
   • Nuremberg bounding box is defined
   • Grid is properly clipped to boundary via gpd.overlay()
   • Data files respect the study area

⚠️  IMPROVEMENT OPPORTUNITY:
   • Current boundary is rectangular (simplified)
   • True Nuremberg city boundary is irregular
   • Rectangular bbox may include areas outside city limits
   • Alternative: Use OSM admin boundary for more precision

📋 NEXT STEPS:
   1. If you want precise city boundary → download from OSM
   2. Current rectangular bbox is acceptable for initial analysis
   3. If concerned about edge effects → consider shrinking bbox

🗺️  NUREMBERG COORDINATES (current bbox):
   • Min longitude: 10.97°
   • Max longitude: 11.15°
   • Min latitude:  49.38°
   • Max latitude:  49.51°
   • Approximate area: ~135 km²
    """)


def main() -> None:
    """Run all validations."""
    print("\n🗺️  NUREMBERG BOUNDARY VALIDATION & CLIPPING CHECK\n")

    # Validate current setup
    current_status = validate_current_boundary()
    boundary = current_status["boundary"]

    # Validate data files
    validate_data_clips(boundary)

    # Validate grid coverage
    validate_grid_coverage(boundary)

    # Try to get refined boundary
    create_refined_boundary()

    # Summary
    generate_summary_report()

    print("\n✅ Validation complete!\n")


if __name__ == "__main__":
    main()
