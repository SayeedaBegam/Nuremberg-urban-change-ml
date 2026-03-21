# Data Acquisition Guide: 2016-2018 Temporal Expansion

## Summary

To expand your dataset from 2020-2021 to include 2016-2018, you need to acquire:

- **Sentinel-2 L2A imagery** (June 2016, 2017, 2018)
- **ESA WorldCover labels** (2016, 2017, 2018)

Both datasets are **publicly available but require manual download** due to external API and access restrictions.

---

## Option 1: Manual Download (Recommended)

This is the most straightforward approach for 2-3 years of data.

### Step 1: Download Sentinel-2 Imagery

**Source:** ESA Copernicus Open Access Hub
**URL:** https://scihub.copernicus.eu/

1. Go to https://scihub.copernicus.eu/ (create free account if needed)
2. Click "Search" and filter:
   - **Product type:** S2MSL2A
   - **Date:** 2016-06-01 to 2016-06-30 (adjust for other years)
   - **Geometry:** Draw box around Nuremberg or upload `nuremberg_boundary.geojson`
   - **Cloud coverage:** 0-50%
3. Sort by "Ingestion date" (newest first)
4. Download the product with lowest cloud cover
5. Extract to: `data/raw/S2A_MSIL2A_*.SAFE/`

**Repeat for 2017 and 2018**

### Step 2: Download ESA WorldCover

**Source:** ESA WorldCover Portal
**URL:** https://esa-worldcover.org/

1. Go to https://esa-worldcover.org/maps/
2. Select year (2016, 2017, or 2018)
3. Click on Nuremberg area or download for your region
4. Download GeoTIFF (300-400 MB per year)
5. Save as: `data/raw/ESA_WorldCover_10m_YYYY_Map.tif`

**Alternative:** Download via Python with STAC:

```python
# Install: pip install stac-client requests
from stacfinder import search_stac
results = search_stac(
    area="Nuremberg",
    start_date="2016-01-01",
    end_date="2016-12-31",
    collections=["esa-worldcover"]
)
# Download results
```

---

## Option 2: Automated Download (Google Earth Engine)

If you have a Google Earth Engine account:

```python
import ee
ee.Initialize()

# Define Nuremberg boundary
nuremberg = ee.Geometry.Rectangle([10.97, 49.38, 11.15, 49.51])

# Sentinel-2
s2 = ee.ImageCollection("COPERNICUS/S2_SR") \
    .filterBounds(nuremberg) \
    .filterDate("2016-06-01", "2016-06-30") \
    .filter(ee.Filter.lt("CLOUDS", 50)) \
    .first()

# Export
task = ee.batch.Export.image.toDrive(
    image=s2,
    region=nuremberg,
    scale=10,
    folder="nuremberg-data"
)
task.start()

# WorldCover
wc = ee.Image("ESA/WorldCover/v100/2016") \
    .clip(nuremberg)

task = ee.batch.Export.image.toDrive(
    image=wc,
    region=nuremberg,
    scale=10,
    folder="nuremberg-data"
)
task.start()
```

---

## Option 3: Process Locally Downloaded Files

Once you've manually downloaded the data:

### Sentinel-2

The downloaded SAFE directory from Copernicus contains all required bands. Use our pipeline directly:

```bash
# Just place in data/raw/ and the discovery function will find it
mv S2A_MSIL2A_20160615*.SAFE data/raw/
```

### WorldCover

If downloaded GeoTIFF needs clipping/processing:

```bash
python scripts/download_worldcover.py \
    --input /path/to/downloaded/ESA_WorldCover_10m_2016_Map.tif \
    --year 2016 \
    --output-dir data/raw/
```

---

## Step 3: Build Datasets

Once all data is in `data/raw/`:

```bash
# Verify data is present
ls data/raw/ | grep -E "S2.*SAFE|worldcover"

# Build datasets
python -m src.data.prepare_dataset --years 2016 2017 2018
```

---

## Expected File Structure

```
data/raw/
├── nuremberg_boundary.geojson          (already have)
├── worldcover_t1.tif                   (already have - 2020)
├── worldcover_t2.tif                   (already have - 2021)
├── S2A_MSIL2A_20200615*.SAFE/          (already have)
├── S2A_MSIL2A_20210615*.SAFE/          (already have)
│
├── S2A_MSIL2A_20160615*.SAFE/          (↓ DOWNLOAD THESE ↓)
├── S2A_MSIL2A_20170615*.SAFE/
├── S2A_MSIL2A_20180615*.SAFE/
├── ESA_WorldCover_10m_2016_Map.tif
├── ESA_WorldCover_10m_2017_Map.tif
└── ESA_WorldCover_10m_2018_Map.tif
```

---

## Output After Processing

```bash
python -m src.data.prepare_dataset --years 2016 2017 2018
```

Generates:

```
data/processed/
├── dataset_2016.csv                     (7,290 rows × 25 cols)
├── dataset_2017.csv
├── dataset_2018.csv
├── change_dataset_2016_2017.csv         (7,290 rows × 54 cols)
├── change_dataset_2017_2018.csv
└── [existing 2020-2021 files untouched]
    ├── dataset_2020.csv
    ├── dataset_2021.csv
    └── change_dataset_2020_2021.csv
```

---

## Troubleshooting

### "Sentinel-2 SAFE product for 2016 not found"

**Solution:** Check file location and naming:
```bash
ls -la data/raw/S2*SAFE*
# Should show something like: S2A_MSIL2A_20160615T102031_N0214_R065_T32UQD_*.SAFE
```

### "ESA_WorldCover_10m_2016_Map.tif not found"

**Solution:** Process downloaded file:
```bash
# If you have ESA_WorldCover_10m_2016_some_other_name.tif
python scripts/download_worldcover.py \
    --input data/raw/ESA_WorldCover_10m_2016_*.tif \
    --year 2016
```

### Network timeout / 403 Forbidden

This is expected when trying automated downloads from external servers. Use manual download instead.

---

## Alternative: Use Mock Data for Testing

If you want to test the pipeline without real data:

```bash
# Generate synthetic test data for 2016-2018
python scripts/create_mock_data.py --years 2016 2017 2018

# Then run pipeline
python -m src.data.prepare_dataset --years 2016 2017 2018
```

---

## Data Statistics

| Source | Year | Availability | Size |
|--------|------|--------------|------|
| Sentinel-2 L2A | 2016-2021 | ✓ Daily | ~8 GB/month per region |
| ESA WorldCover | 2015-2020 | ✓ Annual | 300-400 MB per year |
| ESA WorldCover | 2021+ | ✗ Not yet | - |

---

## Next Steps

1. ✓ Download Sentinel-2 for 2016, 2017, 2018 (from Copernicus Hub)
2. ✓ Download WorldCover for 2016, 2017, 2018 (from ESA website)
3. ✓ Place files in `data/raw/`
4. Run: `python -m src.data.prepare_dataset --years 2016 2017 2018`

**Estimated time:** 2-3 hours (mostly waiting for downloads)

Once you have the data, reach out and we can process it!
