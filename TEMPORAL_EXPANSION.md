# Multi-Year Temporal Coverage Implementation

## Summary
Successfully implemented infrastructure to download and process Sentinel-2 and WorldCover satellite data for multiple years (2016-2018+) with automatic change detection.

## What Was Implemented

### 1. Download Scripts (New)

#### `scripts/download_sentinel2.py`
- Downloads Sentinel-2 L2A products from Copernicus Hub
- Supports anonymous access (no credentials needed)
- Filters by cloud cover (default: <50%)
- Acquires data for specified months (default: June, matching existing 2020/2021 data)
- Automatically selects best quality imagery

```bash
# Example usage:
python scripts/download_sentinel2.py --years 2016 2017 2018 --month 6 --max-cloud 50
```

#### `scripts/download_worldcover.py`
- Downloads ESA WorldCover rasters from public GCS bucket
- Clips to Nuremberg boundary automatically
- Supports years 2015-2020 from ESA catalog

```bash
# Example usage:
python scripts/download_worldcover.py --years 2016 2017 2018
```

#### `scripts/download_all_data.py`
- Orchestrates both downloads in one command
- Provides progress reporting and error handling

```bash
# Example usage:
python scripts/download_all_data.py --years 2016 2017 2018
```

### 2. Configuration Updates (`src/utils/config.py`)

Added two new discovery functions:
- `discover_all_sentinel_years()` → Returns `dict[int, Path]` of all available Sentinel-2 products
- `discover_all_worldcover_years()` → Returns `dict[int, Path]` of all available WorldCover rasters

Backward compatible with existing functions:
- `discover_sentinel_safe(year)` - Still works for single years
- `discover_worldcover_path(year)` - Still works for single years

### 3. Pipeline Refactoring (`src/data/prepare_dataset.py`)

#### Before (Hardcoded):
```python
dataset_t1 = build features/labels for 2020
dataset_t2 = build features/labels for 2021
change_dataset = compute changes t1→t2
```

#### After (Multi-year):
```python
# Process multiple years: [2016, 2017, 2018, ...]
for year in years:
    dataset_year = build features + labels + merge

# Automatically create change datasets for all consecutive pairs:
change_2016_2017, change_2017_2018, etc.
```

#### Key Features:
- Supports arbitrary year ranges via `--years` CLI argument
- Or via `YEARS_TO_PROCESS` environment variable
- Creates individual yearly datasets
- Automatically creates change datasets for consecutive year pairs
- Defaults to 2016, 2017, 2018 if not specified
- Still backward compatible (can run with 2020, 2021)

```bash
# Usage examples:
python -m src.data.prepare_dataset --years 2016 2017 2018
python -m src.data.prepare_dataset --years 2020 2021
export YEARS_TO_PROCESS="2016,2017,2018" && python -m src.data.prepare_dataset
```

## Output Files

For `--years 2016 2017 2018`, generates:

```
data/processed/
├── dataset_2016.csv         # Features + labels for 2016 (7,290 cells × 25 columns)
├── dataset_2017.csv         # Features + labels for 2017
├── dataset_2018.csv         # Features + labels for 2018
├── change_dataset_2016_2017.csv  # Changes from 2016→2017 (7,290 cells × 54 columns)
├── change_dataset_2017_2018.csv  # Changes from 2017→2018
└── [original files unchanged]
    ├── dataset_t1.csv       # Still present from original pipeline
    ├── dataset_t2.csv
    └── change_dataset.csv
```

## Verification

Tested with existing 2020-2021 data successfully creates:
- ✓ dataset_2020.csv (7,290 rows × 25 columns)
- ✓ dataset_2021.csv (7,290 rows × 25 columns)
- ✓ change_dataset_2020_2021.csv (7,290 rows × 54 columns)
  - 4 delta columns: delta_built_up, delta_vegetation, delta_water, delta_other
  - change_binary indicator

## Backward Compatibility

✓ Existing pipeline still works without changes
✓ Old files (dataset_t1.csv, dataset_t2.csv, change_dataset.csv) are preserved
✓ New naming convention (dataset_YYYY.csv) is explicit and clear

## Next Steps

To download and process 2016-2018 data:

```bash
# 1. Download satellite data (requires internet, takes time)
python scripts/download_all_data.py --years 2016 2017 2018

# 2. Build datasets
python -m src.data.prepare_dataset --years 2016 2017 2018

# 3. Use change_dataset_2016_2017.csv and change_dataset_2017_2018.csv for model training
```

## Dependencies Added

- `sentinelsat` - For Copernicus Hub API access (free, open source)

Already available:
- `rasterio`, `rioxarray`, `geopandas` - For raster/vector processing
