"""Generate synthetic Sentinel-2 and WorldCover data for testing the pipeline."""
import os
from pathlib import Path
import tempfile
import shutil

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import box

# Nuremberg bounding box (4326)
BBOX_4326 = (10.97, 49.38, 11.15, 49.51)
WIDTH, HEIGHT = 512, 512  # Mock raster dimensions

def create_mock_worldcover(year: int, output_path: Path) -> None:
    """Create a synthetic WorldCover raster."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic raster data with WorldCover classes
    # Classes: 10=tree, 20=shrub, 30=herbaceous, 40=moss, 50=built-up, 60=barren, 80=water, etc.
    data = np.random.choice([10, 20, 30, 50, 60, 80], size=(HEIGHT, WIDTH), p=[0.25, 0.15, 0.1, 0.3, 0.15, 0.05])
    
    # Transform for the bounding box
    transform = from_bounds(*BBOX_4326, WIDTH, HEIGHT)
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=HEIGHT,
        width=WIDTH,
        count=1,
        dtype=data.dtype,
        transform=transform,
        crs=CRS.from_epsg(4326),
    ) as dst:
        dst.write(data, 1)
    
    print(f"✓ Created synthetic WorldCover for {year}: {output_path}")

def create_mock_sentinel_safe(year: int, output_dir: Path) -> None:
    """Create a synthetic Sentinel-2 SAFE directory structure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create SAFE directory name
    safe_name = f"S2A_MSIL2A_{year}0615T102031_N0214_R065_T32UQD_20{year}0615T102026.SAFE"
    safe_path = output_dir / safe_name
    safe_path.mkdir(exist_ok=True)
    
    # Create required subdirectories
    granule_path = safe_path / "GRANULE" / "L2A_T32UQD_A000001_20200615T102026" / "IMG_DATA" / "T32UQD_20200615T102031_B"
    granule_path.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic band files
    bands = {
        "B02_10m": 512,
        "B03_10m": 512,
        "B04_10m": 512,
        "B08_10m": 512,
        "B11_20m": 256,
        "B12_20m": 256,
    }
    
    transform_10m = from_bounds(*BBOX_4326, 512, 512)
    transform_20m = from_bounds(*BBOX_4326, 256, 256)
    
    for band_name, size in bands.items():
        band_path = granule_path / f"{band_name}.jp2"
        data = np.random.randint(0, 10000, size=(size, size), dtype=np.uint16)
        transform = transform_10m if "10m" in band_name else transform_20m
        
        with rasterio.open(
            band_path,
            'w',
            driver='JP2OpenJPEG',
            height=size,
            width=size,
            count=1,
            dtype=data.dtype,
            transform=transform,
            crs=CRS.from_epsg(4326),
        ) as dst:
            dst.write(data, 1)
    
    print(f"✓ Created synthetic Sentinel-2 SAFE for {year}: {safe_path}")

def create_mock_boundary(output_path: Path) -> None:
    """Create a synthetic Nuremberg boundary GeoJSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a simple bounding box
    geom = box(*BBOX_4326)
    gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[geom],
        crs="EPSG:4326"
    )
    gdf.to_file(output_path, driver="GeoJSON")
    print(f"✓ Created synthetic boundary: {output_path}")

def main():
    # Get the project root correctly
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent  # Go up from scripts/ to project root
    raw_dir = project_root / "data" / "raw"
    
    print("Creating synthetic test data...")
    print(f"Output directory: {raw_dir}\n")
    
    # Create WorldCover rasters
    create_mock_worldcover(2020, raw_dir / "worldcover_t1.tif")
    create_mock_worldcover(2021, raw_dir / "worldcover_t2.tif")
    
    # Create Sentinel-2 SAFE directories
    create_mock_sentinel_safe(2020, raw_dir)
    create_mock_sentinel_safe(2021, raw_dir)
    
    # Create boundary
    create_mock_boundary(raw_dir / "nuremberg_boundary.geojson")
    
    print("\n✅ All synthetic data created successfully!")
    print("\nYou can now run:")
    print("  python -m src.data.prepare_dataset")

if __name__ == "__main__":
    main()
