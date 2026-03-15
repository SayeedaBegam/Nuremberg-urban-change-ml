# Mapping Urban Change in Nuremberg with Machine Learning

This repository contains the final assignment project for **Machine Learning WT 25/26** at **UTN**.  
The project builds a tabular machine learning system that predicts **land-cover composition** and **land-cover change** in Nuremberg from satellite imagery and ESA WorldCover labels, then presents the results in a small interactive dashboard.

## 1. Assignment Goal

The assignment asks for a model-based system that:

- uses **satellite imagery** for at least two time periods
- uses **ESA WorldCover** as the main source of land-cover labels
- works with **tabular features**, not CNNs or end-to-end computer vision
- predicts **land-cover composition** and **land-cover change**
- evaluates models beyond plain accuracy
- communicates **uncertainty**, **limitations**, and **trust** to non-experts
- provides a working **interactive product**

This repository was built specifically around those requirements.

## 2. Project Framing

### Study area

- City: **Nuremberg, Germany**
- Geometry source:
  - preferred: Nuremberg boundary file
  - fallback used in code: a tighter Nuremberg bounding box if the exact boundary is missing

### Temporal setup

- `t1`: **2020**
- `t2`: **2021**

### Spatial unit

- default grid size: **250m x 250m**
- configurable through environment variable `GRID_SIZE_METERS`

This was chosen because:

- 10m pixel-level modeling is too noisy for a one-day tabular ML project
- aggregated grid cells are easier to explain and evaluate
- grid-level outputs fit the assignment’s focus on interpretable, responsible ML

### Intended users

- non-expert city stakeholders
- students or public users exploring broad urban change patterns

### Non-intended use

The system must **not** be used for:

- parcel-level planning
- legal or regulatory decisions
- property valuation
- causal claims about why a specific place changed

## 3. What We Predict

For each grid cell, the system derives:

### Land-cover composition labels

- `built_up_prop`
- `vegetation_prop`
- `water_prop`
- `other_prop`

### Land-cover change labels

- `delta_built_up`
- `delta_vegetation`
- `delta_water`
- `delta_other`

An additional `change_binary` label is derived as a simple helper target for change/no-change interpretation.

## 4. Data Used

### Mandatory data

1. **Sentinel-2 L2A imagery**
   - one product for **2020**
   - one product for **2021**
   - used to engineer spectral and index-based tabular features

2. **ESA WorldCover**
   - **2020**
   - **2021**
   - used as the main label source

### Optional bonus data

3. **OpenStreetMap context features**
   - optional `.gpkg` or `.shp`
   - used for:
     - `road_density`
     - `building_density`

The OSM part is implemented as an optional extension and does not block the main workflow.

## 5. How the Pipeline Works

The project is intentionally structured as a simple end-to-end pipeline.

### Step 1: Build the study grid

- load Nuremberg geometry
- create a regular grid over the study area
- compute centroids for spatial splitting and mapping

### Step 2: Build satellite features

From Sentinel-2, the code extracts and aggregates:

- band means and standard deviations for:
  - `B02`
  - `B03`
  - `B04`
  - `B08`
  - `B11`
  - `B12`
- spectral indices:
  - `NDVI`
  - `NDBI`
  - `NDWI`
- brightness summary

These are fixed-length tabular features, which satisfies the assignment constraint against CNN-style models.

### Step 3: Build labels from ESA WorldCover

WorldCover classes are grouped into:

- `built_up`
- `vegetation`
- `water`
- `other`

The code computes class proportions inside each grid cell for each year.

### Step 4: Create the change dataset

The `2020` and `2021` tables are joined by `cell_id`, then the code computes year-to-year deltas.

### Step 5: Train two models

Required by the assignment:

- **Elastic Net**
  - interpretable baseline
- **Random Forest**
  - nonlinear tabular model

### Step 6: Evaluate beyond accuracy

The code includes:

- a **spatial hold-out split**
- **MAE**
- **RMSE**
- **false change rate**
- **stability score**
- a **stress test** under noisy input features

### Step 7: Serve the results in a dashboard

A **Streamlit** app displays:

- the Nuremberg grid map
- predicted change layers
- uncertainty
- explanation text
- limitations text

## 6. Repository Structure

```text
Final_asmt/
  data/
    raw/
    interim/
    processed/
  models/
  reports/
  src/
    app/
    data/
    models/
    utils/
  README.md
  requirements.txt
```

### Important folders

- `src/data`
  - data ingestion, grid construction, feature engineering, label generation
- `src/models`
  - training, evaluation, uncertainty
- `src/app`
  - Streamlit dashboard
- `reports`
  - report scaffold, reflection notes, checklist, demo script

## 7. Raw Data Expectations

The code supports either clean filenames or the original download names.

### Supported inputs

- `data/raw/nuremberg_boundary.geojson`
- `data/raw/worldcover_t1.tif`
- `data/raw/worldcover_t2.tif`
- `data/raw/sentinel_t1.tif`
- `data/raw/sentinel_t2.tif`

or the original downloaded forms:

- `ESA_WorldCover_10m_2020_*_Map.tif`
- `ESA_WorldCover_10m_2021_*_Map.tif`
- Sentinel-2 `.SAFE` directories for `2020` and `2021`

### Optional bonus input

- an OSM `.gpkg` or `.shp` containing roads and/or buildings

## 8. Setup and Run

Run everything from the project root:

`c:\Users\sayee\Documents\UTN_Sem3\ML\Final_asmt`

### Create and activate environment

```powershell
C:\Users\sayee\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv
.\.venv\Scripts\activate.bat
.\.venv\Scripts\python -m pip install -r requirements.txt
```

### Prepare the dataset

```powershell
.\.venv\Scripts\python -m src.data.prepare_dataset
```

### Train the models

```powershell
.\.venv\Scripts\python -m src.models.train_all
```

### Launch the dashboard

```powershell
.\.venv\Scripts\streamlit run src/app/app.py
```

### Optional: change grid resolution

```powershell
$env:GRID_SIZE_METERS="250"
.\.venv\Scripts\python -m src.data.prepare_dataset
```

## 9. Main Outputs

### Processed data

- `data/processed/dataset_t1.csv`
- `data/processed/dataset_t2.csv`
- `data/processed/change_dataset.csv`
- `data/processed/app_predictions.csv`

### Interim geometry

- `data/interim/grid.geojson`

### Trained models

- `models/elastic_net.joblib`
- `models/random_forest.joblib`
- `models/metrics.json`

## 10. How This Matches the Assignment

### Problem framing and scope

Covered by:

- this `README`
- `reports/final_report_outline.md`
- `reports/assignment_checklist.md`

### Data exploration and reality check

The project is designed to discuss at least these issues:

- seasonal and acquisition effects
- WorldCover label noise
- spatial autocorrelation
- mixed land-cover cells
- possible geometry mismatch between imagery and labels

One issue intentionally not fully fixed:

- label noise at class boundaries and mixed cells

### Modeling

Satisfied by:

- `ElasticNet`
- `RandomForestRegressor`

Both operate on tabular features only.

### Evaluation beyond accuracy

Satisfied by:

- spatial hold-out split
- regression metrics
- false change rate
- stability score
- feature-noise stress test

### Explainability and trust

Satisfied by:

- helpful explanation text in the app
- misleading explanation example in the app
- uncertainty estimate from tree disagreement
- explicit limitations panel

### Interactive product

Satisfied by:

- Streamlit dashboard in `src/app/app.py`

### Mandatory ChatGPT reflection

Prepared in:

- `reports/chatgpt_reflection.md`
- `reports/chatgpt_usage_log.md`

## 11. Key Implementation Decisions

These are the main project choices and why they were made.

### Why tabular features instead of images directly?

Because the assignment explicitly disallows CNNs and wants fixed-length features.

### Why 2020 and 2021?

Because ESA WorldCover is clearly available for both years and aligns well with the project scope.

### Why 250m grid cells?

Because this is a practical balance between:

- too much noise at pixel level
- too much smoothing at very coarse levels
- runtime constraints for a one-day build

### Why Elastic Net and Random Forest?

Because together they provide:

- one interpretable baseline
- one stronger nonlinear model

That directly satisfies the assignment.

## 12. Known Limitations

- Predictions are only as reliable as the input label quality.
- WorldCover is not perfect ground truth.
- Temporal mismatch can still exist between image acquisition dates and yearly land-cover labels.
- The fallback Nuremberg bounding box is broader than an exact administrative boundary.
- OSM features, if used, are static context features and not true time-aligned annual observations.
- The dashboard supports broad interpretation, not precise operational decisions.

## 13. Files for Submission Support

- `reports/final_report_outline.md`
- `reports/assignment_checklist.md`
- `reports/chatgpt_reflection.md`
- `reports/chatgpt_usage_log.md`
- `reports/demo_video_script.md`

## 14. Current Status

At this stage, the repository supports:

- real data ingestion from your downloaded Sentinel-2 and WorldCover files
- processed dataset generation
- model training
- metric export
- dashboard prediction export
- Streamlit app launch

The remaining work for a final polished submission is mainly:

- report writing
- dashboard polishing
- screenshots and demo recording
- optional OSM bonus integration
