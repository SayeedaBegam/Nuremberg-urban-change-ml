# Mapping Urban Change in Nuremberg with Machine Learning

## Short Overview

This repository contains a final project for **Machine Learning WT 25/26 (UTN)** on land-cover modeling in **Nuremberg, Germany**.

The project is a **tabular machine learning** system built around:

- Sentinel-2-derived fixed-length features
- ESA WorldCover-based land-cover labels
- multi-output regression for land-cover composition
- derived land-cover change analysis
- a file-based **Streamlit dashboard** for visualization, uncertainty, error analysis, interpretability, and stress testing

The repository currently contains two related tracks:

- an **original scaffold pipeline** for change-target modeling from processed geospatial tables
- a newer **temporal composition workflow** with:
  - `Sentinel-2` (`ee`)
  - `Sentinel-2 + OSM` (`ee_osm`)

## Problem Statement

Urban areas change over time, but those changes are not always well captured by simple single-class maps. This project models land-cover composition at the grid-cell level in Nuremberg and uses those composition predictions to support change analysis.

Instead of predicting one dominant class, the main temporal workflow predicts four land-cover proportions per cell:

- `built_up`
- `vegetation`
- `water`
- `other`

This keeps the problem aligned with the assignment requirement to use **tabular / fixed-length features only**, without CNNs, Transformers, or end-to-end image models.

## Project Objectives

- Build a tabular ML pipeline for Nuremberg using satellite-derived predictors and ESA WorldCover labels.
- Predict **T2 land-cover composition** from **T1 features**.
- Support **derived land-cover change analysis** from those composition predictions.
- Compare multiple classical ML models.
- Evaluate both in-period performance and harder forward temporal generalization.
- Provide an interactive dashboard with:
  - maps
  - actual vs predicted composition views
  - uncertainty and error summaries
  - model comparison
  - interpretability and stress-test views

## Repository Structure

```text
.
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── processed_esa_ee/
│   ├── processed_esa_ee_osm/
│   ├── train_esa_ee/
│   └── train_esa_osm/
├── models/
│   ├── elastic_net_ee/
│   ├── random_forest_ee/
│   ├── xgboost_ee/
│   ├── mlp_ee/
│   ├── elastic_net_ee_osm/
│   ├── random_forest_ee_osm/
│   ├── xgboost_ee_osm/
│   ├── mlp_ee_osm/
│   └── model_comparison_ee_osm/
├── reports/
├── src/
│   ├── app/
│   ├── data/
│   ├── models/
│   └── utils/
├── README.md
└── requirements.txt
```

Key folders:

- `src/app/`: Streamlit app, map helpers, dashboard utilities, stress-test helper
- `src/data/`: original dataset preparation pipeline
- `src/models/`: model training, evaluation, export utilities
- `src/utils/`: configuration and data-preparation helpers
- `data/processed_esa_ee/`: processed Sentinel-2 / WorldCover temporal tables
- `data/processed_esa_ee_osm/`: processed Sentinel-2 + OSM temporal tables
- `data/train_esa_ee/`: temporal training tables for the Sentinel-2-only pipeline
- `data/train_esa_osm/`: temporal training tables for the Sentinel-2 + OSM pipeline
- `models/`: trained model artifacts, metrics, prediction exports, comparison outputs

## Data Sources

### Main data sources

- **Sentinel-2**-derived tabular predictors for at least two time periods
- **ESA WorldCover** labels for land-cover composition
- optional **OSM-derived context features** for the `ee_osm` workflow

### Files currently present in the repository

Examples of processed files already available in this branch:

- `data/train_esa_ee/nuremberg_2019_to_2020_model.csv`
- `data/train_esa_ee/nuremberg_2020_to_2021_model.csv`
- `data/train_esa_osm/nuremberg_2019_to_2020_model_osm.csv`
- `data/train_esa_osm/nuremberg_2020_to_2021_model_osm.csv`
- `data/processed_esa_ee/nuremberg_2019_features_250m.csv`
- `data/processed_esa_ee/nuremberg_2020_composition_250m.csv`
- `data/processed_esa_ee/nuremberg_2021_composition_250m.csv`
- `data/processed_esa_ee_osm/nuremberg_grid_250m_wgs84.geojson`

The repository also contains raster/vector inputs used by the earlier scaffold workflow, including raw WorldCover and boundary files.

## Method / Pipeline Overview

### Study area and spatial unit

- **Study area**: Nuremberg, Germany
- **Spatial unit**: stable **250 m x 250 m grid cells**
- **Identifier**: `cell_id`

### Main temporal composition workflow

```text
Processed Sentinel-2 / ESA WorldCover tables
    -> temporal train table construction
    -> T1 feature selection
    -> T2 composition labels
    -> train / validation / test split on 2019 -> 2020
    -> forward evaluation on 2020 -> 2021
    -> prediction exports + metrics
    -> Streamlit dashboard
```

### Learning setup

- **Task type**: multi-output regression
- **Input**: T1 tabular features only
- **Output**: T2 land-cover composition
- **Outputs**:
  - `built_up`
  - `vegetation`
  - `water`
  - `other`

### Temporal setup

For the main temporal experiments:

- `2019 -> 2020` is used for **train / validation / test**
- `2020 -> 2021` is used for **forward external evaluation only**

### Output post-processing

Predictions are constrained to valid proportions:

- clipped to `[0, 1]`
- row-normalized so the outputs sum to `1`

In the OSM-based workflow, three targets are predicted directly and `other` is derived during post-processing, but the final outputs still form a four-part composition.

## Current Models / Experiments Implemented

### Original scaffold change pipeline

Files:

- `src/data/prepare_dataset.py`
- `src/models/train_all.py`

This older path builds a change dataset and trains:

- Elastic Net
- Random Forest

for **change targets** such as `delta_built_up`, `delta_vegetation`, `delta_water`, `delta_other`.

### Sentinel-2 temporal composition pipeline (`ee`)

Training scripts:

- `src/models/train_elastic_net_ee.py`
- `src/models/train_random_forest_ee.py`
- `src/models/train_xgboost_ee.py`
- `src/models/train_mlp_ee.py`

### Sentinel-2 + OSM temporal composition pipeline (`ee_osm`)

Training scripts:

- `src/models/train_elastic_net_ee_osm.py`
- `src/models/train_random_forest_ee_osm.py`
- `src/models/train_xgboost_ee_osm.py`
- `src/models/train_mlp_ee_osm.py`

### Model families currently implemented

- Elastic Net
- Random Forest
- XGBoost
- MLP

## App / Visualization Overview

The current dashboard is implemented in:

- `src/app/app.py`

It is a **file-based dashboard**, not a live inference service.

### Current app behavior

The app can switch between two data-source views:

- `Sentinel-2`
- `Sentinel-2 + OSM`

It currently supports:

- model selection
- split selection (`test_2019_2020`, `forward_2020_2021`)
- composition vs change view
- class selection
- predicted and actual map panels
- uncertainty summaries
- error analysis
- model comparison across splits
- Elastic Net coefficient view
- feature-noise stress test

### Current dashboard inputs

The app reads artifacts from files such as:

- `data/processed/app_predictions.csv`
- `models/metrics.json`
- `models/*_ee/*.json`
- `models/*_ee_osm/*.json`
- `models/*_ee/*.csv`
- `models/*_ee_osm/*.csv`
- `data/interim/grid.geojson`
- `data/processed_esa_ee_osm/nuremberg_grid_250m_wgs84.geojson`

### Notes on current status

- The top row of the dashboard is map-based and compares predicted vs actual composition where available.
- Change-based views depend on the availability of T1 composition for the selected split.
- Some diagnostics are derived from regression outputs rather than native classification outputs.

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Optional dependency

`xgboost` is used by the XGBoost training scripts, but it is not listed in `requirements.txt` in this branch. Install it manually if you want to train or run the XGBoost pipelines:

```bash
pip install xgboost
```

## How to Run

### 1. Run the original scaffold pipeline

```bash
python -m src.data.prepare_dataset
python -m src.models.train_all
```

### 2. Train the Sentinel-2 temporal models

```bash
python -m src.models.train_elastic_net_ee
python -m src.models.train_random_forest_ee
python -m src.models.train_xgboost_ee
python -m src.models.train_mlp_ee
```

### 3. Build app-ready exports for the Sentinel-2 baseline view

```bash
python -m src.models.export_app_artifacts
```

### 4. Train the Sentinel-2 + OSM temporal models

```bash
python -m src.models.train_elastic_net_ee_osm
python -m src.models.train_random_forest_ee_osm
python -m src.models.train_xgboost_ee_osm
python -m src.models.train_mlp_ee_osm
```

### 5. Optional OSM comparison summary

```bash
python -m src.models.compare_models_ee_osm
```

### 6. Launch the dashboard

```bash
streamlit run src/app/app.py
```

## Current Status / Roadmap

### Current status

Implemented in this branch:

- original change-target scaffold pipeline
- temporal composition pipelines for `Sentinel-2` and `Sentinel-2 + OSM`
- four classical model families
- saved model artifacts, metrics, prediction exports, and comparison files
- a Streamlit dashboard with multiple analytical sections

### Roadmap / planned improvements

The repository still has room for improvement, including:

- tighter documentation consistency between the older scaffold and the newer temporal workflow
- clearer treatment of missing T1 composition for some split-specific change views
- additional reporting and packaging polish for final submission

## Limitations

- The project is restricted to **tabular fixed-length features**; it does not use dense image models.
- Forward generalization is harder than in-period test performance and should be interpreted carefully.
- Some dashboard change views are **derived** from composition outputs rather than directly learned as classification labels.
- Some dashboard sections depend on the availability of specific artifacts or T1 composition tables.
- The dashboard is intended for **broad exploratory interpretation**, not parcel-level or operational decision-making.
- The repository contains both an older scaffold and a newer temporal workflow, so not every script serves the same prediction objective.

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- geopandas
- rasterio / rioxarray
- Streamlit
- Folium / streamlit-folium
- joblib
- matplotlib / seaborn
- optional: xgboost

## Authors / Course Context

- Project: **Mapping Urban Change in Nuremberg with Machine Learning**
- Context: **Machine Learning WT 25/26**
- Institution: **UTN**

Supporting course-deliverable files in this repository include:

- `reports/assignment_checklist.md`
- `reports/final_report_outline.md`
- `reports/chatgpt_reflection.md`
- `reports/chatgpt_usage_log.md`
- `reports/demo_video_script.md`
