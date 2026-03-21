# Nuremberg Urban Change Analysis: 2020 → 2021
## Real Data Analysis Complete ✅

---

## What You Have

### Raw Data (2020-2021)
- ✅ **Sentinel-2 imagery** (June 2020 & 2021) - 12 spectral bands
- ✅ **ESA WorldCover labels** (2020 & 2021) - Land cover classification

### Processed Data (7,290 grid cells)
1. **dataset_2020.csv** (7,290 rows × 25 cols)
   - Spectral features (B2-B12 means/stds)
   - Spectral indices (NDVI, NDBI, NDWI, brightness)
   - Land cover proportions (built-up, vegetation, water, other)

2. **dataset_2021.csv** (7,290 rows × 25 cols)
   - Same structure as 2020

3. **change_dataset_2020_2021.csv** (7,290 rows × 54 cols)
   - All features from 2020 (with _t1 suffix)
   - All features from 2021 (with _t2 suffix)
   - **Delta columns** (the changes!)
   - Binary change indicator

### Change Analysis Summary
**Total area changes: 53.03% of Nuremberg** (3,866 / 7,290 cells)

---

## What Actually Changed (2020 → 2021)

### 📈 Built-up Area (Urbanization)
- **1,255 cells** with significant increase (>10% growth)
- **1,186 cells** with significant decrease
- **Net change**: +0.12% (slight urbanization)
- **Most intense**: cell_07238 went from **20% → 80% built-up**

### 🌳 Vegetation
- **1,385 cells** lost vegetation
- **1,354 cells** gained vegetation
- **Net change**: -0.08% (slight loss)
- **Most extreme loss**: cell_07244 lost 66.7% vegetation
- **Most extreme gain**: cell_07211 gained 83.3% vegetation

### 💧 Water
- **1,216 cells** gained water
- **1,228 cells** lost water
- **Mostly balanced** (natural variation)

---

## Exported Analysis Files

All files saved in `data/processed/`:

### 1. **urbanization_hotspots_2020_2021.csv** (216 cells)
```
Top 5 Rapid Urbanization Cells:
  1. cell_07238: 20% → 80% (+60%)
  2. cell_07203: 16.7% → 66.7% (+50%)
  3. cell_07284: 0% → 50% (+50%)
  4. cell_07216: 50% → 100% (+50%)
  5. cell_07225: 0% → 50% (+50%)
```
**Use for**: Identifying development zones, urban planning analysis

### 2. **vegetation_loss_hotspots_2020_2021.csv** (303 cells)
```
Top 5 Most Vegetation Loss:
  1. cell_07244: 83.3% → 16.7% (-66.7%)
  2. cell_07216: 50% → 0% (-50%)
  3. cell_04080: 77.1% → 31.4% (-45.7%)
  4. cell_01102: 70% → 26.7% (-43.3%)
  5. cell_05625: 75% → 33.3% (-41.7%)
```
**Use for**: Environmental impact, deforestation tracking

### 3. **greenification_hotspots_2020_2021.csv** (285 cells)
```
Top 5 Most Vegetation Gain:
  1. cell_07211: 0% → 83.3% (+83.3%)
  2. cell_07214: 20% → 80% (+60%)
  3. cell_07246: 0% → 60% (+60%)
  4. cell_07201: 33.3% → 83.3% (+50%)
  5. cell_07213: 33.3% → 83.3% (+50%)
```
**Use for**: Green urban projects, reforestation initiatives

### 4. **change_magnitude_2020_2021.csv** (7,290 cells)
- All cells ranked by total change magnitude
- Combines all land cover changes into single score
**Use for**: Machine learning feature engineering

### 5. **change_summary_2020_2021.csv**
- Summary statistics
- Total hotspot counts
- Mean changes by land cover type

---

## What You Can Do Next

### Option 1: Train a Change Detection Model 🤖
```bash
# Use change_dataset_2020_2021.csv to train ML model
# Predict where changes are likely to happen
# Features: All spectral + indices from 2020
# Target: change_binary (1 = changed, 0 = no change)
```

### Option 2: Visualize Changes on Map 🗺️
```bash
# Use urbanization_hotspots_2020_2021.csv
# Use vegetation_loss_hotspots_2020_2021.csv
# Use greenification_hotspots_2020_2021.csv
# Create heat maps showing change intensity
```

### Option 3: Temporal Analysis (2019-2020-2021) ⏰
Currently have: 2020 → 2021
Could add: 2019 → 2020 → 2021
(Need to acquire/generate 2019 data)

### Option 4: Explanatory Analysis 📊
```bash
# Use change_binary as target
# Use features_t1 to explain what predicts change
# Questions: "What makes an area likely to urbanize?"
```

---

## How to Use These Files

### For Machine Learning:
```python
import pandas as pd

# Load change data
change_df = pd.read_csv('data/processed/change_dataset_2020_2021.csv')

# Features: All 52 columns except change_binary and cell_id
X = change_df.drop(['cell_id', 'change_binary', 'year_label_t1', 'year_label_t2'], axis=1)

# Target: Binary change indicator
y = change_df['change_binary']

# Train model (Random Forest, XGBoost, etc.)
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()
# model.fit(X, y)
```

### For Visualization:
```python
# Load hotspots
urban_df = pd.read_csv('data/processed/urbanization_hotspots_2020_2021.csv')
veg_df = pd.read_csv('data/processed/vegetation_loss_hotspots_2020_2021.csv')

# Get cell geometry from grid
from src.data.load_boundary import load_boundary
grid = gpd.read_file('data/interim/grid.geojson')

# Merge with hotspots and visualize
# Plot as heat map with change magnitude as color
```

### For Further Analysis:
```python
# All 5 exported CSVs available in data/processed/
ls data/processed/change_*.csv
# - change_dataset_2020_2021.csv (main)
# - change_summary_2020_2021.csv
# - change_magnitude_2020_2021.csv
# - urbanization_hotspots_2020_2021.csv
# - vegetation_loss_hotspots_2020_2021.csv
# - greenification_hotspots_2020_2021.csv
```

---

## Quick Reference

### Commands to Run
```bash
# Analyze changes
python -m scripts.analyze_change --year1 2020 --year2 2021

# Export hotspots
python -m scripts.export_change_summary --year1 2020 --year2 2021

# Build change detection model (in your existing ML pipeline)
python -m src.data.prepare_dataset --years 2020 2021
```

### Key Files
- `dataset_2020.csv` - 2020 features & labels (2.9 MB)
- `dataset_2021.csv` - 2021 features & labels (2.9 MB)
- `change_dataset_2020_2021.csv` - Full change matrix (6.2 MB)
- `*.csv` hotspots - Sorted change cells (ready for visualization)

### Grid Information
- **Grid cells**: 7,290
- **Cell size**: 250m × 250m
- **Grid file**: `data/interim/grid.geojson`
- **CRS**: EPSG:3857 (Web Mercator)
- **Coverage**: Nuremberg city boundary

---

## What's Interesting About Your Data

### Strong Signals:
✅ **53% of city changed** - High change detection rate
✅ **Clear hotspots** - 216 rapid urbanization zones, 303 vegetation loss zones
✅ **Balanced trade-offs** - Urban growth paired with vegetation shifts
✅ **Extreme cases** - Some cells 100% built-up (cell_07216), others 100% vegetated (cell_07221)

### Ready for ML:
✅ **7,290 labeled samples** - Good training data
✅ **25 features per year** - Rich feature set
✅ **Binary target** - Clear change/no-change labels
✅ **Spectral diversity** - 12 bands capture land cover well

---

## Next Steps (Your Choice)

1. **Train a change detector** → Predict 2021-2022 changes
2. **Visualize hot spots** → Map urbanization and vegetation loss
3. **Extend timeline** → Add 2019 data for longer temporal analysis
4. **Explain changes** → Which features drive urbanization?
5. **Policy application** → Identify planning zones for intervention

What would you like to focus on? 🚀
